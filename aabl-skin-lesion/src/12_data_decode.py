# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 1.3  Decode once at 224x224 (as the paper: resize to 224), keep uint8 on the GPU
# =============================================================================
def decode(paths, size):
    out = np.empty((len(paths), size, size, 3), np.uint8)

    def one(i):
        with Image.open(paths[i]) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
    with ThreadPoolExecutor(max_workers=min(32, 2 * (os.cpu_count() or 4))) as ex:
        list(tqdm(ex.map(one, range(len(paths))), total=len(paths), desc="decode", mininterval=2))
    return out


def to_gpu(arr):
    g = torch.empty((len(arr), 3, arr.shape[1], arr.shape[2]), dtype=torch.uint8, device=DEVICE)
    for a in range(0, len(arr), 512):
        g[a:a + 512] = torch.from_numpy(arr[a:a + 512]).to(DEVICE).permute(0, 3, 1, 2)
    return g


cache_npy = os.path.join(OUT, f"cache_{CFG.IMG_SIZE}_{hashlib.md5(''.join(LABELS.image_id).encode()).hexdigest()[:8]}.npy")
if os.path.exists(cache_npy):
    X = np.load(cache_npy)
else:
    X = decode([IMG_IDX[i.lower()] for i in LABELS.image_id], CFG.IMG_SIZE); np.save(cache_npy, X)
GX = to_gpu(X); GY = torch.from_numpy(y_all).long()
_s, _s2, _n = np.zeros(3), np.zeros(3), 0                 # train mean / std in chunks (ISIC 2019 is ~24k images)
for a in range(0, len(SPLIT["train"]), 1024):
    _c = X[SPLIT["train"][a:a + 1024]].astype(np.float64) / 255
    _s += _c.sum((0, 1, 2)); _s2 += (_c ** 2).sum((0, 1, 2)); _n += _c.shape[0] * _c.shape[1] * _c.shape[2]
MEAN = torch.tensor(_s / _n, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(np.sqrt(_s2 / _n - (_s / _n) ** 2), dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
del _c, X
GX_T = GY_T = None
if TEST_LABELS is not None:
    GX_T = to_gpu(decode([TEST_IDX[i.lower()] for i in TEST_LABELS.image_id], CFG.IMG_SIZE))
    GY_T = torch.from_numpy(TEST_LABELS.y.values).long()
print("train mean", MEAN.flatten().cpu().numpy().round(4), "std", STD.flatten().cpu().numpy().round(4))


class Loader:
    """GPU batches. train: RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(ROTATION)
    (black fill, as torchvision), then normalisation. Randomness depends on (seed, epoch) only,
    so within a seed every module sees the identical batch order and augmentation."""

    def __init__(self, gx, gy, idx, seed, bs, train):
        self.gx, self.gy = gx, gy
        self.idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        self.seed, self.bs, self.train, self.epoch = int(seed), int(bs), train, 0

    def __len__(self):
        n = len(self.idx)
        return n // self.bs if self.train else -(-n // self.bs)

    def _norm(self, x):
        return ((x - on_cur(MEAN)) / on_cur(STD)).contiguous(memory_format=MEMFMT)

    def __iter__(self):
        if not self.train:
            for a in range(0, len(self.idx), self.bs):
                j = self.idx[a:a + self.bs]
                yield self._norm(on_cur(self.gx)[j.to(on_cur(self.gx).device)].float().div_(255)), self.gy[j]
            return
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch); self.epoch += 1
        perm = self.idx[torch.randperm(len(self.idx), generator=g)]
        for b in range(len(self)):
            j = perm[b * self.bs:(b + 1) * self.bs]; n = len(j)
            fx = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
            fy = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
            a = (torch.rand(n, generator=g) * 2 - 1) * CFG.ROTATION * math.pi / 180
            th = torch.zeros(n, 2, 3)
            th[:, 0, 0] = fx * a.cos(); th[:, 0, 1] = -fx * a.sin()
            th[:, 1, 0] = fy * a.sin(); th[:, 1, 1] = fy * a.cos()
            x = on_cur(self.gx)[j.to(on_cur(self.gx).device)].float().div_(255)
            grid = F.affine_grid(th.to(x.device), list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            yield self._norm(x), self.gy[j]


# nested lesion-grouped, class-stratified fractions of the TRAINING split for RQ4 (25 % ⊂ 50 % ⊂ 100 %; fixed)
_tr = SPLIT["train"]
_q = [te for _, te in (StratifiedGroupKFold(4, shuffle=True, random_state=CFG.SPLIT_SEED + 1)
                       .split(np.zeros(len(_tr)), y_all[_tr], LABELS.lesion.values[_tr] if "lesion" in LABELS else np.arange(len(_tr))))]
TRAIN_FRACTION = {1.0: _tr, 0.5: np.sort(_tr[np.concatenate(_q[:2])]), 0.25: np.sort(_tr[_q[0]])}
assert set(TRAIN_FRACTION[0.25]) <= set(TRAIN_FRACTION[0.5]) <= set(TRAIN_FRACTION[1.0])


def loaders(seed, fraction=1.0):
    return (Loader(GX, GY, TRAIN_FRACTION[fraction], seed, CFG.BATCH_SIZE, True),
            Loader(GX, GY, SPLIT["val"], seed, 128, False),
            Loader(GX, GY, SPLIT["test"], seed, 128, False))


_a = next(iter(loaders(0)[0])); _b = next(iter(loaders(0)[0]))
assert torch.equal(_a[0], _b[0]), "loader not reproducible"
print(f"batch {tuple(_a[0].shape)} | same seed -> identical batch [ok]")
