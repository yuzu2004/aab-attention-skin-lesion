# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.6  RQ4 — Boundary conditions: AAB-L - NA gap vs training-set size (25 / 50 / 100 %) and pretrained vs scratch
# =============================================================================
#      100 % ImageNet: AAB-L (this run) vs NA (earlier run, same protocol). 25 / 50 % and scratch: both trained here.
if len(R) and SHOW:
    rows = []
    for init, frac in [("ImageNet-1k", f) for f in sorted(FRACTIONS) + [1.0]] + [("scratch", 1.0)]:
        d0 = ALL if (init == "ImageNet-1k" and frac == 1.0) else R[(R.Init == init) & (R.Fraction == frac)]
        for col in (PRIMARY, "BalancedAcc"):
            r = paired("AABL", "NA", col, data=d0)
            if r:
                rows.append(dict(init=init, fraction=frac, NA_mean=d0[d0.Module == "NA"][col].mean(),
                                 AABL_mean=d0[d0.Module == "AABL"][col].mean(), **r))
    if rows:
        B4 = pd.DataFrame(rows)
        B4["verdict"] = [verdict(r) for _, r in B4.iterrows()]
        print("Table 4 — boundary conditions (AAB-L - NA, pp, paired by seed):")
        display(B4.round(4)); B4.to_csv(os.path.join(OUT, "tables", "table4_boundary.csv"), index=False)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for a_, col in zip(ax, (PRIMARY, "BalancedAcc")):
            d = B4[(B4.metric == col) & (B4.init == "ImageNet-1k")].sort_values("fraction")
            if len(d):
                a_.errorbar(100 * d.fraction, d.diff_pp, yerr=[d.diff_pp - d.lo, d.hi - d.diff_pp], marker="o", capsize=4,
                            label="ImageNet init")
            ds = B4[(B4.metric == col) & (B4.init == "scratch")]
            if len(ds):
                a_.errorbar(100 * ds.fraction, ds.diff_pp, yerr=[ds.diff_pp - ds.lo, ds.hi - ds.diff_pp], marker="s",
                            capsize=4, ls="none", label="scratch")
            a_.axhline(0, c="k", lw=.8)
            if np.isfinite(NOISE.get(col, {}).get("noise_floor_pp", np.nan)):
                a_.axhspan(-NOISE[col]["noise_floor_pp"], NOISE[col]["noise_floor_pp"], color="gray", alpha=.15, label="noise floor")
            a_.set_xlabel("% of training data"); a_.set_ylabel("AAB-L - NA (pp)"); a_.set_title(col); a_.legend(fontsize=8)
        plt.tight_layout(); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True); plt.savefig(os.path.join(OUT, "figures", "rq4_boundary.png"), dpi=150); plt.show()
    else:
        print("tier-4 runs not finished yet")
