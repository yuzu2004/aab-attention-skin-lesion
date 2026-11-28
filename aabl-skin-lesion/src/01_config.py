# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 0.2  Configuration   [PAPER PROTOCOL, Table II, ResNet-50, 5 seeds] — only the new AAB-L family is trained here.
#      NA / SE / CBAM / ECA / TA are NOT retrained: their numbers come from the earlier run (section 5).
#      (The config fields below are unchanged so the config hash matches the earlier run; the old AAB is not used.)
# =============================================================================
SMOKE = False      # True -> tiny end-to-end check (a few minutes)


@dataclass(frozen=True)
class Config:
    # ---- protocol (Pedro & Oliveira 2022) ----------------------------------------
    SEEDS: tuple = (0, 1, 2, 3, 4)           # 5 paired seeds (the paper: 3 runs); seeds 0-2 already on disk are reused
    IMG_SIZE: int = 224
    ROTATION: float = 20.0                   # RandomRotation(20)
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LR_SCRATCH: float = 0.1
    LR_PRETRAINED: float = 0.001
    STEP_EPOCHS: int = 30                    # lr x 0.1 every 30 epochs
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 1e-4
    USE_CLASS_WEIGHTS: bool = False          # not stated in the paper -> plain CE

    # ---- which tables to run ------------------------------------------------------
    RUN_TABLE_I: bool = False                # random init, ResNet-34/50/101 x modules (expensive)
    RUN_TABLE_II: bool = True                # ImageNet-1k ResNet-50 x modules
    TABLE_I_BACKBONES: tuple = ("resnet34", "resnet50", "resnet101")
    TABLE_II_BACKBONE: str = "resnet50"
    MODULES: tuple = ("AABL",)               # (kept for the hash; WHAT is trained is set by RUN_TIERS below)

    # ---- proposed AAB -----------------------------------------------------------------
    AAB_GAMMA_INIT: float = 0.2              # gamma = 0 is an absorbing stationary point (6.4)
    AAB_GATE_INIT_STD: float = 0.1
    AAB_CA_HEAD_STD: float = 0.5
    AAB_CA_MODE: str = "auto"                # pool4 | pool4_dw | auto (pool4 if C <= 512 else pool4_dw)
    AAB_REDUCTION: int = 16
    AAB_SA_KERNEL: int = 7
    # ---- AAB fixes found by the section-8 diagnosis (defaults = original AAB) ---------------
    AAB_GAMMA_FIXED: Optional[float] = None  # e.g. 1.0: gamma is a constant -> F*(1 + (2M-1)) = F*2M; the
                                             # gradient on Mc/Ms/gate is no longer multiplied by a small gamma
    AAB_IDENTITY_INIT: bool = False          # Mc = Ms = 0.5 exactly at init (ca.fc2 = 0, sa = 0) -> block = identity
    ATTN_LAYERS: tuple = (1, 2, 3, 4)        # stages that receive the attention module (EVERY module type)
    AAB_CENTER: bool = False                 # True: s = 1 + 2g(M - mu), mu = mean of M over images (no static path)
    AAB_ASYM_LOGIT: float = 0.0              # e.g. 0.85: Mc starts ~0.70, Ms ~0.30 -> the gate has a gradient
    AAB_GATE_DESC: str = "gapstd"            # gate input: "gapstd" = [GAP; std] (2C) | "pool2std" = [avgpool 2x2; std] (5C)

    # ---- training recipe for the NEW (attention) parameters ---------------------------
    #   protocol = 1 / 1 / True / 0 / 0. Any other value is a deviation from the paper and is applied to
    #   the attention parameters of EVERY module (gamma settings exist only in AAB).
    ATTN_LR_MULT: float = 1.0        # lr multiplier for all parameters inside the attention modules
    GAMMA_LR_MULT: float = 1.0      # lr multiplier for AAB gamma
    GAMMA_WD: bool = True            # weight decay on AAB gamma
    COLLAPSE_REG: float = 0.0       # loss -= COLLAPSE_REG * mean((2M-1)^2): rewards a decisive mask
    GAMMA_WARMUP_EPOCHS: int = 0    # gamma frozen during the first N epochs

    # ---- data / split -----------------------------------------------------------------
    DATASET: str = "ham10000"                # "ham10000" (the paper) | "isic2019_7cls" (ISIC 2019, 7 classes of 2018)
    ISIC19_ROOT: str = "/kaggle/input/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification"
    DATA_ROOT: str = ""                      # "" -> search /kaggle/input (or ./data_isic2018)
    SPLIT_SEED: int = 42
    N_SPLIT_FOLDS: int = 7                   # 5 / 1 / 1
    MAX_PER_CLASS: Optional[int] = None
    VAL_EVERY: int = 5                       # monitor val every N epochs (the final model is evaluated)

    # ---- run control ------------------------------------------------------------------
    DETERMINISTIC: bool = False              # True = bit-reproducible but slower convs; not part of the protocol
    CHANNELS_LAST: bool = True
    TIME_BUDGET_H: float = 11.5
    OUT: str = "/kaggle/working/po_aab" if ON_KAGGLE else "./po_aab"


CFG = Config()
if isinstance(CFG.MODULES, str):                        # ("AAB") without a comma is a string, not a tuple
    CFG = replace(CFG, MODULES=(CFG.MODULES,))
# ---- what this notebook trains (tiers run in this order; within a tier seed by seed) -----------------------------
OTHER_SEEDS = (0, 1,2,3,4)       # seeds for tiers 2-4 (exploratory; (0,) = one seed, descriptive only). Tier 1 uses CFG.SEEDS
RUN_TIERS = (1, 2, 3, 4)   # remove a tier to skip it -> 1: AAB-L (main) | 2: capacity-matched ablations of AAB-L | 3: one-lever anti-saddle recipe |
                           # 4: boundary conditions (NA + AAB-L at 25 / 50 % data, and from scratch) — NA is trained
                           #    here ONLY for tier 4 (those settings are not in the earlier run)
FRACTIONS = (0.25, 0.5)    # data-fraction sweep (100 % = tier 1 + earlier NA)
BOUNDARY_MODULES = ("NA", "AABL")
if SMOKE:
    CFG = replace(CFG, SEEDS=(0, 1), EPOCHS=2, STEP_EPOCHS=1, MAX_PER_CLASS=24, BATCH_SIZE=8,
                  IMG_SIZE=64, TABLE_II_BACKBONE="resnet34", MODULES=("AABL",),
                  VAL_EVERY=1, OUT=CFG.OUT + "_smoke")
OUT = os.path.abspath(CFG.OUT)
if CFG.DATASET != "ham10000":
    OUT += "_" + CFG.DATASET
for d in ("weights", "preds", "tables", "history", "figures"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
RES_CSV = os.path.join(OUT, "results.csv")
MEMFMT = torch.channels_last if (CFG.CHANNELS_LAST and USE_AMP) else torch.contiguous_format

# Fields that are at their protocol value are left out of the hash, so runs made before these fields existed
# (same protocol) are still recognised. Any deviation changes the hash and the results are kept apart.
PROTOCOL_DEFAULTS = dict(ATTN_LR_MULT=1.0, GAMMA_LR_MULT=1.0, GAMMA_WD=True, COLLAPSE_REG=0.0,
                         GAMMA_WARMUP_EPOCHS=0, DATASET="ham10000", AAB_GAMMA_FIXED=None,
                         AAB_IDENTITY_INIT=False, ATTN_LAYERS=(1, 2, 3, 4),
                         AAB_GATE_DESC="gapstd", AAB_CENTER=False, AAB_ASYM_LOGIT=0.0)
_skip = {"SEEDS", "OUT", "TIME_BUDGET_H", "RUN_TABLE_I", "RUN_TABLE_II", "VAL_EVERY", "ISIC19_ROOT"}
_SPEED = dict(DETERMINISTIC=True, CHANNELS_LAST=True,               # run-selection switches are
              MODULES=("NA", "SE", "CBAM", "ECA", "TA", "AAB"))    # hashed at fixed values too    # speed switches do not change the protocol:
                                                         # hashed at fixed values so earlier runs still match
_hd = {k: _SPEED.get(k, v) for k, v in asdict(CFG).items()
       if k not in _skip and not (k in PROTOCOL_DEFAULTS and v == PROTOCOL_DEFAULTS[k])}
if CFG.AAB_IDENTITY_INIT:
    _hd["AAB_IDENTITY_INIT_IMPL"] = "fc2_std0.01"          # v1 (fc2 = 0) left fc1 without gradient
CFG_HASH = hashlib.md5(json.dumps(_hd, sort_keys=True, default=str).encode()).hexdigest()[:10]
IS_PROTOCOL = all(getattr(CFG, k) == v for k, v in PROTOCOL_DEFAULTS.items()
                  if k not in ("DATASET", "AAB_GAMMA_FIXED", "AAB_IDENTITY_INIT", "ATTN_LAYERS", "AAB_GATE_DESC",
                               "AAB_CENTER", "AAB_ASYM_LOGIT"))


def read_results():
    """The module label "NA" (no attention, as in the paper) must not be parsed as a missing value."""
    return pd.read_csv(RES_CSV, keep_default_na=False, na_values=[""])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = CFG.DETERMINISTIC
    torch.backends.cudnn.benchmark = not CFG.DETERMINISTIC


print(f"protocol: SGD m={CFG.MOMENTUM} wd={CFG.WEIGHT_DECAY} batch {CFG.BATCH_SIZE} | "
      f"lr {CFG.LR_SCRATCH} (scratch) / {CFG.LR_PRETRAINED} (pretrained), x0.1 every {CFG.STEP_EPOCHS} ep | "
      f"{CFG.EPOCHS} epochs")
print(f"recipe  : {'PAPER PROTOCOL' if IS_PROTOCOL else 'MODIFIED'} | attention lr x{CFG.ATTN_LR_MULT:g} (all modules) | "
      f"AAB gamma init {CFG.AAB_GAMMA_INIT}, lr x{CFG.GAMMA_LR_MULT:g}, wd {'on' if CFG.GAMMA_WD else 'off'}, "
      f"warm-up {CFG.GAMMA_WARMUP_EPOCHS} ep | collapse reg {CFG.COLLAPSE_REG:g}")
print(f"AAB     : gamma {'FIXED at ' + str(CFG.AAB_GAMMA_FIXED) if CFG.AAB_GAMMA_FIXED is not None else 'learned'} | "
      f"identity init {CFG.AAB_IDENTITY_INIT} | centred {CFG.AAB_CENTER} | asym {CFG.AAB_ASYM_LOGIT} | gate input {CFG.AAB_GATE_DESC} | attention in stages {CFG.ATTN_LAYERS} (all modules)")
print(f"data    : {CFG.DATASET} | seeds {CFG.SEEDS} | modules {CFG.MODULES}")
print(f"out {OUT} | config hash {CFG_HASH}" + ("   [SMOKE]" if SMOKE else ""))
_hf = os.path.join(OUT, "cfg_hash.txt")            # weights / predictions are named WITHOUT the hash
if os.path.exists(_hf) and open(_hf).read().strip() != CFG_HASH:
    print(f"WARNING: {OUT} was last used by config {open(_hf).read().strip()}; weights and predictions of the same "
          "run key are overwritten. Use a separate CFG.OUT for any non-protocol configuration.")
open(_hf, "w").write(CFG_HASH)
if not IS_PROTOCOL or CFG.AAB_GAMMA_FIXED is not None or CFG.AAB_IDENTITY_INIT or CFG.AAB_CENTER or CFG.AAB_ASYM_LOGIT:
    print("WARNING: this is NOT the final protocol configuration (see the AAB / recipe lines above).")


# ---- AAB-L (light, routed AAB) — FIXED before running; part of the AAB-L implementation tag (a change -> retrain) --------
AABL_CFG = dict(
    stages=(3, 4),          # AAB-L only in stages 3-4 (9 blocks): fewer blocks = fewer kernel launches = faster
    pool_dw=True,           # Mc descriptor = learned depthwise 4x4 pooling (AAB-Pool lineage, init = plain average) + std
    asym=1.0,               # Mc starts ~ sigmoid(+1) = 0.73, Ms ~ sigmoid(-1) = 0.27 -> gate gradient ~ gamma (Mc - Ms) != 0
    gamma0=0.7,             # gamma = sigmoid(raw), init 0.7 (old AAB: 0.2 clamped + weight decay -> gate gradient ~ 0.2)
    gate_a0=0.55,           # static part of the gate starts at a = 0.55
    router_dim=16,          # router: 4 mask statistics + LayerNorm(Linear(2C -> 16)) -> MLP 20 -> 16 -> 1
    route_start=5, route_ramp=5,   # per-image term off for epochs 1-5, ramps 0 -> 1 over epochs 6-10
    route_scale=1.0,        # R (logit units) of the centred per-image term
    run_momentum=0.1,       # running mean of the router output (used at inference)
    branch_drop=0.2,        # training only: per image, p = 0.2 -> gate forced to 0 or 1 (both branches stay useful)
    attn_adamw=True,        # AAB-L parameters: AdamW (module optimiser); backbone + fc keep the protocol SGD 1e-3
    adamw_lr=dict(mask=2e-3, gate=2e-3, gain=2e-3), adamw_wd=1e-4,   # decayed by the same x0.1 / 30 StepLR
    param_budget=(23.0e6, 25.0e6),  # total ResNet-50 + AAB-L parameters must fall in this range (checked in 3.1)
)
AABL_IMPL = "aabl-v1-" + hashlib.md5(json.dumps(AABL_CFG, sort_keys=True).encode()).hexdigest()[:8]
DISPLAY = {"NA": "NA", "SE": "SE", "CBAM": "CBAM", "ECA": "ECA", "TA": "TA", "AABL": "AAB-L",
           "AABL_SUP": "AAB-L-suppress", "AABL_CONST": "AAB-L-constgate", "AABL_FIX": "AAB-L-fixedgate",
           "AABL_NOSP": "AAB-L-nospatial", "AABL_NAIVE": "AAB-L-naive", "AABL_N_GAMMA": "AAB-L-naive+gamma",
           "AABL_N_INIT": "AAB-L-naive+init", "AABL_N_LR": "AAB-L-naive+lr", "AABL_HARD": "AAB-L-naive-hardclamp"}
print(f"AAB-L   : {AABL_CFG} | impl {AABL_IMPL}")

# ---- resume across Kaggle sessions: copy this notebook's earlier runs from an attached output (Save Version -> Input)
def _ours(r):            # runs trained by THIS notebook: the AAB-L family, and the tier-4 NA runs (fraction / scratch)
    return r.Module.astype(str).str.startswith("AABL") | r.Key.astype(str).str.contains("_f\\d+_s|_scratch_", regex=True)


if ON_KAGGLE and not os.path.exists(RES_CSV):
    for _prev in sorted(glob.glob("/kaggle/input/**/results.csv", recursive=True)):
        _d = os.path.dirname(_prev)
        try:
            _r = pd.read_csv(_prev, keep_default_na=False, na_values=[""])
        except Exception:
            continue
        if not {"Module", "Key"} <= set(_r) or not _ours(_r).any():
            continue
        _r = _r[_ours(_r)]; _r.to_csv(RES_CSV, index=False)
        for _k in _r.Key:
            for _sub, _ext in (("weights", ".pt"), ("history", ".csv"), ("preds", ".npz"), ("preds", "_isic.npz")):
                _src = os.path.join(_d, _sub, _k + _ext)
                if os.path.exists(_src) and not os.path.exists(os.path.join(OUT, _sub, _k + _ext)):
                    shutil.copy(_src, os.path.join(OUT, _sub, _k + _ext))
        print(f"restored {len(_r)} run(s) of this notebook from {_d}")
        break
