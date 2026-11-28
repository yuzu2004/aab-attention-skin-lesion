# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 0.3  Automatic download of the data and the ImageNet weights from Kaggle (kagglehub)
#      Skipped when the data / weights are already present (attached Kaggle inputs, CFG.DATA_ROOT or ./data_isic2018).
#      Environment variables: AABL_AUTO_DOWNLOAD=0 to disable, AABL_DATA_HANDLE / AABL_WEIGHTS_HANDLE to change the source.
# =============================================================================
import subprocess, sys

DATA_HANDLE = os.environ.get("AABL_DATA_HANDLE", "yuw1409/data-isic2018")
WEIGHTS_HANDLE = os.environ.get("AABL_WEIGHTS_HANDLE", "yuw1409/model-resnet/pytorch/default")
AUTO_DOWNLOAD = os.environ.get("AABL_AUTO_DOWNLOAD", "1") == "1"
TORCHVISION_RESNET50 = "https://download.pytorch.org/models/resnet50-0676ba61.pth"   # same file, fallback only
DOWNLOADED_DATA_ROOTS, DOWNLOADED_WEIGHT_DIRS = [], []


def _label_csv_present(base):
    """HAM10000 metadata or ISIC 2018 Task 3 training ground truth somewhere under base"""
    if not base or not os.path.isdir(base):
        return False
    for dp, _, fns in os.walk(base):
        for f in fns:
            lf = f.lower()
            if lf.endswith(".csv") and ("ham10000_metadata" in lf or ("groundtruth" in lf and "test" not in lf)) and "2019" not in lf:
                return True
    return False


def _resnet_weights_present(dirs):
    return any("resnet" in os.path.basename(p).lower()
               for d in dirs if d and os.path.isdir(d) for p in glob.glob(os.path.join(d, "**", "*.pth"), recursive=True))


def _kagglehub():
    try:
        import kagglehub
    except ImportError:
        print("installing kagglehub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        import kagglehub
    return kagglehub


_local_bases = [CFG.DATA_ROOT] if CFG.DATA_ROOT else (["/kaggle/input"] if ON_KAGGLE else ["./data_isic2018"])
if CFG.DATASET != "ham10000":
    print("[download] dataset is not HAM10000 -> no automatic download")
elif any(_label_csv_present(b) for b in _local_bases):
    print(f"[download] data already present under {_local_bases} -> no download")
elif not AUTO_DOWNLOAD:
    print("[download] AABL_AUTO_DOWNLOAD=0 and no local data: step 10 will stop with instructions")
else:
    try:
        _p = _kagglehub().dataset_download(DATA_HANDLE)
        DOWNLOADED_DATA_ROOTS.append(_p)
        print(f"[download] data {DATA_HANDLE} -> {_p}")
    except Exception as e:
        print(f"!! data download from Kaggle failed ({type(e).__name__}: {e}).\n"
              f"   Check the internet connection and that '{DATA_HANDLE}' is public; if Kaggle asks for credentials, create an API "
              f"token (kaggle.com > Settings > API) and set KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json.")

_need_weights = bool(CFG.RUN_TABLE_II)                       # Table II fine-tunes ImageNet weights
_weight_dirs = ["/kaggle/input"] if ON_KAGGLE else ["./weights"]
if _need_weights and not _resnet_weights_present(_weight_dirs) and AUTO_DOWNLOAD:
    try:
        _p = _kagglehub().model_download(WEIGHTS_HANDLE)
        DOWNLOADED_WEIGHT_DIRS.append(_p)
        print(f"[download] ImageNet weights {WEIGHTS_HANDLE} -> {_p}")
    except Exception as e:
        print(f"!! weights download from Kaggle failed ({type(e).__name__}: {e}) -> official torchvision file instead")
        _d = os.path.abspath("./weights"); os.makedirs(_d, exist_ok=True)
        torch.hub.download_url_to_file(TORCHVISION_RESNET50, os.path.join(_d, os.path.basename(TORCHVISION_RESNET50)))
        DOWNLOADED_WEIGHT_DIRS.append(_d)
        print(f"[download] {TORCHVISION_RESNET50} -> {_d}")
