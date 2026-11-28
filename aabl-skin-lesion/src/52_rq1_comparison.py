# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.2  RQ1 — Does AAB-L beat the plain backbone (NA) and SE, CBAM, ECA, TA?  (ISIC-convention ranking table)
#      References: earlier run, same protocol / split / seeds / batch order (NOT retrained). AAB-L: this run.
# =============================================================================
MET = ("BalancedAcc", "MacroF1", "MacroAUC", "QWK")


def _ms(v):
    v = pd.Series(v).dropna().astype(float)
    return f"{v.mean():.3f}±{v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else "—")


# QWK / McNemar need predictions: this run's, and the earlier run's preds/*.npz if its output is attached
def _preds(module, seed, suffix):
    p = _find_file(run_key("II", CFG.TABLE_II_BACKBONE, module, True, seed) + suffix, "preds")
    return np.load(p) if p else None


def _qwk(module, suffix):
    v = []
    for s in CFG.SEEDS:
        z = _preds(module, s, suffix)
        if z is not None:
            v.append(cohen_kappa_score(z["y_true"], z["y_proba"].argmax(1), weights="quadratic"))
    return v


if SHOW and "AABL" in set(NEW_MAIN.Module):
    rows = []
    for _, r in OLD_TABLE.iterrows():
        rows.append(dict(Module=r.Module, Params=f"{r.ParamsM:.2f}M", **{
            "Internal Bal. acc": f"{r.BalancedAcc[0]:.3f}±{r.BalancedAcc[1]:.3f}", "Internal Macro-AUC": f"{r.MacroAUC[0]:.3f}±{r.MacroAUC[1]:.3f}",
            "Internal QWK": _ms(_qwk(r.Module, ".npz")),
            "ISIC Bal. acc": f"{r.isic_BalancedAcc[0]:.3f}±{r.isic_BalancedAcc[1]:.3f}", "ISIC Macro-F1": f"{r.isic_MacroF1[0]:.3f}±{r.isic_MacroF1[1]:.3f}",
            "ISIC Macro-AUC": f"{r.isic_MacroAUC[0]:.3f}±{r.isic_MacroAUC[1]:.3f}", "ISIC QWK": _ms(_qwk(r.Module, "_isic.npz")),
            "Ref. [1] Bal. acc": f"{r.Ref1[0]:.3f}±{r.Ref1[1]:.3f}"}, _k=r.isic_BalancedAcc[0], Source="earlier run"))
    d = NEW_MAIN[NEW_MAIN.Module == "AABL"]
    rows.append(dict(Module="AAB-L", Params=f"{d.Params.iloc[0] / 1e6:.2f}M", **{
        "Internal Bal. acc": _ms(d.BalancedAcc), "Internal Macro-AUC": _ms(d.MacroAUC),
        "Internal QWK": _ms(d.QWK) if "QWK" in d else _ms(_qwk("AABL", ".npz")),
        "ISIC Bal. acc": _ms(d.isic_BalancedAcc), "ISIC Macro-F1": _ms(d.isic_MacroF1), "ISIC Macro-AUC": _ms(d.isic_MacroAUC),
        "ISIC QWK": _ms(d.isic_QWK) if "isic_QWK" in d else _ms(_qwk("AABL", "_isic.npz")), "Ref. [1] Bal. acc": "—"},
        _k=d.isic_BalancedAcc.mean(), Source=f"this run ({len(d)} seeds)"))
    T1 = pd.DataFrame(rows).sort_values("_k", ascending=False).drop(columns="_k")
    T1.insert(0, "Rank", np.arange(1, len(T1) + 1))
    print(f"Table 1 — ranking by ISIC 2018 balanced accuracy (ISIC convention). Internal test {len(SPLIT['test'])} images, "
          f"ISIC 2018 test {len(GY_T) if GY_T is not None else 0}; mean ± SD over 5 seeds, final epoch (protocol). "
          "QWK reads the class index as ordinal (listed in the proposal; interpret with care); '—' = predictions not attached.")
    display(T1); T1.to_csv(os.path.join(OUT, "tables", "table1_rq1.csv"), index=False)
    display(COST.round(2))

    cmp = [r for b in REFS for col in (PRIMARY, "BalancedAcc", "isic_MacroF1", "MacroF1") for r in [paired("AABL", b, col)] if r]
    if cmp:
        C1 = holm(pd.DataFrame(cmp)); C1["verdict"] = [verdict(r) for _, r in C1.iterrows()]
        print("RQ1 paired by seed (pp, AAB-L minus reference; Holm over the 5 references per metric):")
        display(C1.round(4)); C1.to_csv(os.path.join(OUT, "tables", "rq1_paired.csv"), index=False)

    mc = []
    for b in REFS:
        for s in CFG.SEEDS:
            for nm, suf in (("internal", ".npz"), ("isic", "_isic.npz")):
                A, B = _preds("AABL", s, suf), _preds(b, s, suf)
                if A is None or B is None:
                    continue
                y = A["y_true"]; ca, cb = A["y_proba"].argmax(1) == y, B["y_proba"].argmax(1) == y
                n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                mc.append(dict(comparison=f"AAB-L vs {b}", set=nm, seed=s, aabl_only_correct=n01, ref_only_correct=n10,
                               p=sps.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0))
    if mc:
        MC = pd.DataFrame(mc)
        print("McNemar exact test per seed (AAB-L right & reference wrong vs the reverse):")
        display(MC.groupby(["comparison", "set"]).agg(seeds=("p", "size"), median_p=("p", "median"),
                                                      seeds_p_lt_05=("p", lambda x: int((x < .05).sum())),
                                                      aabl_only=("aabl_only_correct", "sum"), ref_only=("ref_only_correct", "sum")))
        MC.to_csv(os.path.join(OUT, "tables", "rq1_mcnemar.csv"), index=False)
    else:
        print("McNemar needs the earlier run's preds/*.npz (attach its output folder)")
    print("Protocol note: AAB-L uses AdamW 2e-3 for ITS OWN parameters (module optimiser); backbone and classifier use "
          "the paper's SGD exactly as every reference module.")
elif "AABL" not in set(NEW_MAIN.Module):
    print("no AAB-L runs on disk yet (run 4.2)")
