# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 0.1  Runtime: imports, device(s), AMP, logging, per-GPU tensor copies
# =============================================================================
import os, math, json, time, glob, random, hashlib, threading, shutil
from dataclasses import dataclass, replace, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import Bottleneck
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.flop_counter import FlopCounterMode
import scipy.stats as sps
try:
    from IPython.display import display
except ImportError:                                    # plain python
    display = print

ON_KAGGLE = os.path.isdir("/kaggle/input")
USE_AMP = torch.cuda.is_available()                    # AMP (fp16 + GradScaler) on GPU, fp32 on CPU
DEVICE = "cuda" if USE_AMP else "cpu"                  # "cuda" = the CURRENT device of the calling thread
NGPU = torch.cuda.device_count() if USE_AMP else 1
T_START = time.time()
_BUILD_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()
_ON_LOCK = threading.Lock()
_ON = {}


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP) if USE_AMP else torch.autocast("cpu", enabled=False)


def on_cur(t):
    """the same tensor on the calling thread's current GPU (one cached copy per GPU; GPU 0 holds the originals)"""
    if not USE_AMP or t.device.type != "cuda" or t.device.index == torch.cuda.current_device():
        return t
    key = (id(t), torch.cuda.current_device())
    with _ON_LOCK:
        if key not in _ON:
            _ON[key] = t.to(f"cuda:{torch.cuda.current_device()}")
        return _ON[key]


def log(msg):
    with _PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')} {threading.current_thread().name}] {msg}", flush=True)


def hours_used():
    return (time.time() - T_START) / 3600


print(f"torch {torch.__version__} | {'Kaggle' if ON_KAGGLE else 'local'} | "
      + (f"{NGPU} GPU(s): {[torch.cuda.get_device_name(i) for i in range(NGPU)]}" if USE_AMP else "CPU only"))
