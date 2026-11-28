# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.1  Integrity guards (proposal: results are NOT reported if a protocol condition fails), per-seed data of every
#      module, seed-noise floor, paired statistics
# =============================================================================
FORCE_REPORT = False          # True: show the tables even if a guard fails (the failure is still printed)

# earlier run (NOT retrained), Table II protocol, mean ± SD over 5 seeds; Ref [1] = Pedro & Oliveira (2022)
OLD_TABLE = pd.DataFrame([
    ("NA",   23.52, (0.671, 0.016), (0.960, 0.002), (0.657, 0.011), (0.683, 0.008), (0.950, 0.003), (0.647, 0.022)),
    ("SE",   26.05, (0.666, 0.015), (0.959, 0.003), (0.642, 0.022), (0.673, 0.021), (0.947, 0.001), (0.622, 0.018)),
    ("CBAM", 26.05, (0.643, 0.021), (0.953, 0.005), (0.606, 0.010), (0.644, 0.013), (0.938, 0.004), (0.595, 0.021)),
    ("ECA",  23.52, (0.675, 0.020), (0.958, 0.003), (0.645, 0.014), (0.676, 0.013), (0.945, 0.003), (0.662, 0.015)),
    ("TA",   23.53, (0.643, 0.029), (0.952, 0.005), (0.651, 0.018), (0.681, 0.019), (0.944, 0.003), (0.644, 0.038)),
], columns=["Module", "ParamsM", "BalancedAcc", "MacroAUC", "isic_BalancedAcc", "isic_MacroF1", "isic_MacroAUC", "Ref1"])
OLD_ISIC_SEEDS = {"NA": (0.6451, 0.6729, 0.6628, 0.6556, 0.6497), "SE": (0.6379, 0.6734, 0.6494, 0.6375, 0.6129),
                  "CBAM": (0.6072, 0.6228, 0.5978, 0.6043, 0.6001), "ECA": (0.6281, 0.6662, 0.6443, 0.6452, 0.6424),
                  "TA": (0.6409, 0.6412, 0.6406, 0.6824, 0.6491)}       # seeds 0-4; means reproduce the table
for _m, _v in OLD_ISIC_SEEDS.items():
    assert abs(np.mean(_v) - OLD_TABLE.set_index("Module").loc[_m, "isic_BalancedAcc"][0]) < 0.0015, _m
REFS = tuple(OLD_ISIC_SEEDS)

# per-seed values of the reference modules: the earlier results.csv if attached (every metric), else ISIC bal. acc only
OLD_RUNS = None
for _p in sorted(glob.glob("/kaggle/input/**/results.csv", recursive=True)):
    try:
        _r = pd.read_csv(_p, keep_default_na=False, na_values=[""])
    except Exception:
        continue
    if {"Module", "Seed", "CfgHash", "Key"} <= set(_r):
        _r = _r[(_r.CfgHash == CFG_HASH) & _r.Module.isin(REFS) & ~_r.Key.str.contains("_f\\d+_s|_scratch_", regex=True)]
        if _r.Module.nunique() >= 3:
            OLD_RUNS = _r; print(f"earlier per-seed results: {_p} ({len(_r)} runs)"); break
if OLD_RUNS is None:
    OLD_RUNS = pd.DataFrame([dict(Module=m, Seed=s, isic_BalancedAcc=v) for m, vs in OLD_ISIC_SEEDS.items()
                             for s, v in enumerate(vs)])
    print("earlier results.csv not attached: reference modules are paired on ISIC balanced accuracy only")

R = read_results() if os.path.exists(RES_CSV) else pd.DataFrame(columns=["Module", "Key"])
if len(R):
    if "Fraction" not in R:
        R["Fraction"] = 1.0
    R["Fraction"] = R.Fraction.fillna(1.0)
    R = R[R.Module.isin(list(VARIANTS) + list(BOUNDARY_MODULES)) & (R.CfgHash == CFG_HASH)
          & (R.Impl.fillna("") == R.Module.map(impl_tag))].copy()
NEW_MAIN = R[(R.Init == "ImageNet-1k") & (R.Fraction == 1.0) & R.Module.isin(list(VARIANTS))] if len(R) else R
ALL = pd.concat([OLD_RUNS.assign(Source="earlier run"), NEW_MAIN.assign(Source="this run")], ignore_index=True)

guards = {}
if "lesion" in LABELS:
    gs = {k: set(LABELS.lesion.values[v]) for k, v in SPLIT.items()}
    guards["zero lesion overlap train / val / test"] = not (gs["train"] & gs["val"] or gs["train"] & gs["test"] or gs["val"] & gs["test"])
else:
    guards["zero lesion overlap train / val / test"] = False
guards["equal training budget (epochs x steps) within init x fraction"] = bool(
    len(R) == 0 or "TrainSteps" not in R or (R.dropna(subset=["TrainSteps"]).groupby(["Init", "Fraction"]).TrainSteps.nunique() <= 1).all())
guards["capacity-matched controls >= AAB-L parameters"] = bool(CAPACITY_OK)
_plan = [v for v in VARIANTS if VARIANTS[v]["tier"] in RUN_TIERS and VARIANTS[v]["tier"] in (1, 2, 3)]
_cnt = NEW_MAIN.groupby("Module").Seed.nunique() if len(NEW_MAIN) else pd.Series(dtype=int)
_nseed = lambda v: len(CFG.SEEDS) if VARIANTS[v]["tier"] == 1 else len(OTHER_SEEDS)     # planned seeds per variant
missing = {DISPLAY[v]: int(_nseed(v) - _cnt.get(v, 0)) for v in _plan if _cnt.get(v, 0) < _nseed(v)}
guards["all seeds finished for every planned AAB-L variant"] = not missing
display(pd.Series(guards, name="passed").to_frame())
if missing:
    print("missing seeds:", missing)
REPORT_OK = all(guards.values())
SHOW = REPORT_OK or FORCE_REPORT
if not REPORT_OK:
    print("!! a protocol guard failed: the tables are NOT final" + ("" if FORCE_REPORT else " and are not shown (FORCE_REPORT = True to inspect)"))

PRIMARY = "isic_BalancedAcc"


def paired(a, b, col, data=None):
    """a - b by seed, pp: mean, SD, 95% t-CI, wins, paired t p-value, Wilcoxon p (n >= 5)"""
    d0 = ALL if data is None else data
    if col not in d0:
        return None
    A = d0[d0.Module == a].set_index("Seed")[col].dropna(); B = d0[d0.Module == b].set_index("Seed")[col].dropna()
    ss = sorted(set(A.index) & set(B.index))
    if len(ss) < 1:
        return None
    d = 100 * (A.loc[ss].astype(float).values - B.loc[ss].astype(float).values)
    if len(ss) == 1:                                             # one seed: the difference only, no inference
        return dict(comparison=f"{DISPLAY[a]} - {DISPLAY[b]}", metric=col, n=1, diff_pp=d[0], sd_pp=np.nan,
                    lo=np.nan, hi=np.nan, wins=int(d[0] > 0), p_t=np.nan, p_wilcoxon=np.nan)
    h = sps.t.ppf(.975, len(ss) - 1) * d.std(ddof=1) / np.sqrt(len(ss))
    p = sps.ttest_rel(A.loc[ss].astype(float), B.loc[ss].astype(float)).pvalue if d.std() > 0 else 1.0
    pw = sps.wilcoxon(d).pvalue if len(ss) >= 5 and np.any(d != 0) else np.nan
    return dict(comparison=f"{DISPLAY[a]} - {DISPLAY[b]}", metric=col, n=len(ss), diff_pp=d.mean(), sd_pp=d.std(ddof=1),
                lo=d.mean() - h, hi=d.mean() + h, wins=int((d > 0).sum()), p_t=p, p_wilcoxon=pw)


def holm(df, pcol="p_t", by="metric"):
    df = df.copy(); df["p_holm"] = np.nan
    for _, idx in df.groupby(by).groups.items():
        idx = [i for i in idx if np.isfinite(df.loc[i, pcol])]     # Holm only over comparisons with a p-value
        if not idx:
            continue
        p = df.loc[idx, pcol].values; o = np.argsort(p); m = len(p)
        adj = np.maximum.accumulate([min(1.0, (m - i) * p[j]) for i, j in enumerate(o)])
        df.loc[np.asarray(idx)[o], "p_holm"] = adj
    return df


# ---- seed-noise floor (proposal): pooled within-module SD over seeds (every 100 % ImageNet module with >= 3 seeds)
#      -> smallest paired mean difference distinguishable with n seeds (conservative: assumes no seed correlation)
NOISE = {}
for col in (PRIMARY, "BalancedAcc", "isic_MacroF1"):
    if col not in ALL:
        continue
    sds = ALL.groupby("Module")[col].agg(lambda x: x.dropna().std(ddof=1) if x.notna().sum() >= 3 else np.nan).dropna()
    if len(sds):
        n = len(CFG.SEEDS); pooled = 100 * float(np.sqrt((sds ** 2).mean()))
        NOISE[col] = dict(modules=len(sds), pooled_seed_sd_pp=pooled,
                          noise_floor_pp=sps.t.ppf(.975, n - 1) * pooled * np.sqrt(2) / np.sqrt(n))
if NOISE:
    print("seed-noise floor (pp). A difference is reported as REAL only if its paired 95% CI excludes 0 AND |mean| > "
          "noise_floor_pp; reference modules come from a different run, so cross-run rerun noise is added on top:")
    display(pd.DataFrame(NOISE).T.round(3))
NF = NOISE.get(PRIMARY, {}).get("noise_floor_pp", np.nan)


def verdict(r):
    if r["n"] == 1:
        return "1 seed: descriptive only (no inference)"
    if r["n"] < len(CFG.SEEDS):
        return f"exploratory ({int(r['n'])} seeds): " + _verdict(r)
    return _verdict(r)


def _verdict(r):
    nf = NOISE.get(r["metric"], {}).get("noise_floor_pp", NF)
    nf = nf if np.isfinite(nf) else 0.0
    if r["lo"] > 0 and r["diff_pp"] > nf:
        return "better (beyond noise)"
    if r["hi"] < 0 and -r["diff_pp"] > nf:
        return "worse (beyond noise)"
    return "tie (within noise)"
