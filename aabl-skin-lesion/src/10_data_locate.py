# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 1.1  Locate the training data (HAM10000 = the paper, or ISIC 2019 restricted to the 7 ISIC 2018 classes)
#      and, if attached, the official ISIC 2018 Task 3 TEST set (external test)
# =============================================================================
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def image_index(root):
    return {os.path.splitext(f)[0].lower(): os.path.join(dp, f)
            for dp, _, fns in os.walk(root) for f in fns
            if f.lower().endswith(IMG_EXT) and "_mask" not in f.lower()}


def id_column(df):
    cols = {c.lower(): c for c in df.columns}
    return next((cols[k] for k in ("image_id", "image", "image_name", "isic_id") if k in cols), None)


def find_label_csvs(base, test=False):
    keys = ("ham10000_metadata", "groundtruth", "ground_truth")
    out = []
    for dp, _, fns in os.walk(base):
        for f in fns:
            lf = f.lower()
            if not lf.endswith(".csv") or not any(k in lf for k in keys) or "2019" in lf:
                continue
            if ("test" in lf) != test or "validation" in lf:
                continue
            out.append(os.path.join(dp, f))
    return sorted(out, key=len)


def read_labels(csv):
    """DataFrame[image_id, label] from HAM10000 metadata (dx) or an ISIC one-hot ground truth."""
    df = pd.read_csv(csv); icol = id_column(df)
    if icol is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    if "dx" in cols:
        out = pd.DataFrame({"image_id": df[icol].astype(str), "label": df[cols["dx"]].astype(str).str.lower()})
    else:
        num = [c for c in df.columns if c != icol and pd.api.types.is_numeric_dtype(df[c])]
        if len(num) < 2 or not df[num].isin([0, 1, 0.0, 1.0]).all().all():
            return None
        out = pd.DataFrame({"image_id": df[icol].astype(str),
                            "label": np.array(num)[df[num].values.argmax(1)]})
        out["label"] = out["label"].str.lower()
    out["image_id"] = out["image_id"].map(lambda s: os.path.splitext(s)[0])
    return out


def find_lesion_map(base):
    for dp, _, fns in os.walk(base):
        for f in fns:
            if not f.lower().endswith(".csv") or "2019" in f.lower():
                continue
            try:
                df = pd.read_csv(os.path.join(dp, f), nrows=5)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            if "lesion_id" in cols and id_column(df) is not None:
                full = pd.read_csv(os.path.join(dp, f))
                return dict(zip(full[id_column(full)].astype(str).str.lower().map(lambda s: os.path.splitext(s)[0]),
                                full[cols["lesion_id"]].astype(str))), os.path.join(dp, f)
    return None, None


def locate(test=False):
    bases = [CFG.DATA_ROOT] if CFG.DATA_ROOT else (["/kaggle/input"] if ON_KAGGLE else ["./data_isic2018"])
    bases = bases + [d for d in globals().get("DOWNLOADED_DATA_ROOTS", []) if d not in bases]     # step 05 (kagglehub)
    for base in bases:
        if not os.path.isdir(base):
            continue
        for csv in find_label_csvs(base, test=test):
            lab = read_labels(csv)
            if lab is None or len(lab) < 100:
                continue
            ids = set(lab.image_id.str.lower())
            d = os.path.dirname(csv)
            while True:
                idx = image_index(d)
                if len(ids & set(idx)) >= 0.9 * len(ids):
                    return lab, idx, csv
                parent = os.path.dirname(d)
                if parent == d or os.path.abspath(d) == os.path.abspath(base):
                    break
                d = parent
    if test:
        return None, None, None
    raise RuntimeError("HAM10000 / ISIC 2018 Task 3 not found: let step 05 download it (internet + kagglehub), add a dataset "
                       "with HAM10000_metadata.csv (or the Task 3 training GroundTruth CSV) and the images, or set CFG.DATA_ROOT.")


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]          # ISIC 2018 Task 3 classes
NUM_CLASSES = len(CLASS_NAMES)
CLS = {c: i for i, c in enumerate(CLASS_NAMES)}

if CFG.DATASET == "ham10000":
    LABELS, IMG_IDX, LABEL_CSV = locate()
    LABELS = LABELS[LABELS.image_id.str.lower().isin(IMG_IDX) & LABELS.label.isin(CLS)].reset_index(drop=True)
    lesion_map, LESION_CSV = find_lesion_map("/kaggle/input" if ON_KAGGLE and LABEL_CSV.startswith("/kaggle/input")
                                             else os.path.dirname(LABEL_CSV))
    cover = 0.0
    if lesion_map:
        LABELS["lesion"] = LABELS.image_id.str.lower().map(lesion_map)
        cover = LABELS.lesion.notna().mean()
        LABELS["lesion"] = LABELS.lesion.fillna(LABELS.image_id)
    note = ""
elif CFG.DATASET == "isic2019_7cls":
    # ISIC 2019 column -> ISIC 2018 class. 2019 "AK" is actinic keratosis only; 2018 "akiec" also held Bowen's
    # disease, which 2019 files under SCC. SCC and UNK are dropped, so akiec here = AK.
    MAP19 = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}
    LABEL_CSV = os.path.join(CFG.ISIC19_ROOT, "ISIC_2019_Training_GroundTruth.csv")
    LESION_CSV = os.path.join(CFG.ISIC19_ROOT, "ISIC_2019_Training_Metadata.csv")
    gt = pd.read_csv(LABEL_CSV); icol = id_column(gt)
    allc = [c for c in list(MAP19) + ["SCC", "UNK"] if c in gt.columns]
    assert all(c in gt.columns for c in MAP19), f"unexpected ISIC 2019 columns: {list(gt.columns)}"
    c19 = np.array(allc)[gt[allc].values.argmax(1)]
    LABELS = pd.DataFrame({"image_id": gt[icol].astype(str).map(lambda s: os.path.splitext(s)[0]), "c19": c19})
    n_all = len(LABELS); dropped = {c: int((LABELS.c19 == c).sum()) for c in ("SCC", "UNK")}
    LABELS = LABELS[LABELS.c19.isin(MAP19)].copy(); LABELS["label"] = LABELS.c19.map(MAP19)
    IMG_IDX = image_index(CFG.ISIC19_ROOT)
    LABELS = LABELS[LABELS.image_id.str.lower().isin(IMG_IDX)].reset_index(drop=True)
    meta = pd.read_csv(LESION_CSV); mcol = id_column(meta)
    lmap = dict(zip(meta[mcol].astype(str).str.lower(), meta["lesion_id"])) if "lesion_id" in meta else {}
    LABELS["lesion"] = LABELS.image_id.str.lower().map(lmap)
    cover_true = LABELS.lesion.notna().mean()
    LABELS["lesion"] = LABELS.lesion.fillna(LABELS.image_id).astype(str)
    cover = 1.0      # every image has a group key (its lesion_id, or itself) -> grouped split in 1.2
    note = (f"ISIC 2019: {n_all} images, dropped {dropped}; {100 * cover_true:.1f} % of kept images have a "
            "lesion_id, the others form their own group")
else:
    raise ValueError(CFG.DATASET)
LABELS["y"] = LABELS.label.map(CLS)

# official ISIC 2018 Task 3 TEST set (the set Pedro & Oliveira were scored on), if attached
TEST_LABELS, TEST_IDX, TEST_CSV = locate(test=True)
if TEST_LABELS is not None:
    TEST_LABELS = TEST_LABELS[TEST_LABELS.image_id.str.lower().isin(TEST_IDX)
                              & TEST_LABELS.label.isin(CLS)].reset_index(drop=True)
    TEST_LABELS["y"] = TEST_LABELS.label.map(CLS)
    clash = LABELS.image_id.str.lower().isin(set(TEST_LABELS.image_id.str.lower()))
    if clash.any():
        print(f"!! removed {int(clash.sum())} official-test images from the training pool")
        LABELS = LABELS[~clash].reset_index(drop=True)

if CFG.MAX_PER_CLASS:
    LABELS = (LABELS.sample(frac=1.0, random_state=0).groupby("y").head(CFG.MAX_PER_CLASS)
              .sort_index().reset_index(drop=True))          # keeps every column on all pandas versions

print(f"labels   : {LABEL_CSV}")
print(f"lesions  : {LESION_CSV} ({100 * cover:.1f} % of images grouped)" + (f"\n{note}" if note else ""))
print(f"{len(LABELS)} images, {NUM_CLASSES} classes: {CLASS_NAMES}")
print(f"official ISIC 2018 test set: "
      + (f"{len(TEST_LABELS)} images ({TEST_CSV})" if TEST_LABELS is not None else "not found (internal split only)"))
display(LABELS.label.value_counts().rename("images").to_frame().T)
