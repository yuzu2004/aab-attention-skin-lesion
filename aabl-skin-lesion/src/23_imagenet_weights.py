# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 2.4  Local ImageNet weights: each file is matched to its architecture by CONTENT
#      (key names + tensor shapes), so a mislabelled file name cannot load the wrong network.
# =============================================================================
WEIGHT_DIR = "/kaggle/input/models/yuw1409/model-resnet/pytorch/default/1"
WEIGHT_FILES = [os.path.join(WEIGHT_DIR, f) for f in
                ("resnet34-b627a593.pth", "resnet50-0676ba61.pth", "resnet101-63fe2227.pth")]
WEIGHT_FILES = [p for p in WEIGHT_FILES if os.path.exists(p)]
for _wd in ["/kaggle/input", "./weights"] + list(globals().get("DOWNLOADED_WEIGHT_DIRS", [])):    # + step 05 downloads
    WEIGHT_FILES += [p for p in sorted(glob.glob(os.path.join(_wd, "**", "*.pth"), recursive=True))
                     if "resnet" in p.lower() and p not in WEIGHT_FILES]

_SKIP = "num_batches_tracked"        # absent from the official torchvision files; strict load tolerates it


def _sig(sd):
    return {k: tuple(v.shape) for k, v in sd.items() if not k.endswith(_SKIP)}


_REF = {name: _sig(fn(weights=None).state_dict()) for name, fn in BACKBONES.items()}


def _load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


LOCAL_SD.clear()
for path in WEIGHT_FILES:
    size = os.path.getsize(path) / 2 ** 20
    try:
        sd = _load_sd(path)
    except Exception as e:
        print(f"skip    : {path} ({size:.1f} MB) cannot be read: {type(e).__name__}: {e}"); continue
    hit = next((n for n, ref in _REF.items() if _sig(sd) == ref), None)
    if hit is None:
        print(f"unknown : {path} ({size:.1f} MB) matches no torchvision ResNet -> ignored"); continue
    if hit in LOCAL_SD:
        print(f"dup     : {path} ({size:.1f} MB) is another {hit} -> ignored"); continue
    BACKBONES[hit](weights=None).load_state_dict(sd, strict=True)          # real strict check
    LOCAL_SD[hit] = sd
    named = next((b for b in sorted(BACKBONES, key=len, reverse=True)
                  if b in os.path.basename(path).lower()), None)
    flag = "" if named in (None, hit) else f"   <-- FILE NAME SAYS {named}, CONTENT IS {hit}"
    print(f"matched : {hit:10s} <- {os.path.basename(path)} ({size:.1f} MB){flag}")

_need = {CFG.TABLE_II_BACKBONE} if CFG.RUN_TABLE_II else set()        # Table I is from scratch: no weights
print(f"\nlocal ImageNet weights: {sorted(LOCAL_SD)}")
assert not (_need - set(LOCAL_SD)), f"missing ImageNet weights for {sorted(_need - set(LOCAL_SD))}"
print("[ok] ImageNet weights verified by content")
