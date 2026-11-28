# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.5  RQ3 — What does the trained gate do: static recalibration or image-specific routing?
#   level 1: gate SD over images > 0 | level 2: replacing the learned gate by its mean / shuffling it across images
#   costs accuracy (bootstrap CI > 0, consistent over seeds) | level 3: AAB-L beats AAB-L-constgate over seeds (Table 2)
# =============================================================================
if SHOW and len(PH):
    G = PH
    rows = []
    for (v, s), d in G.groupby(["Module", "Seed"]):
        for st in ("val", "isic"):
            for iv in ("mean", "shuffle", "a1", "a0", "off"):
                c = f"rq3_{st}_{iv}_gain_pp"
                if c in d and d[c].notna().any():
                    rows.append(dict(variant=DISPLAY[v], seed=s, set=st, intervention=iv, gain_pp=float(d[c].iloc[0]),
                                     lo=float(d[f"rq3_{st}_{iv}_lo"].iloc[0]), hi=float(d[f"rq3_{st}_{iv}_hi"].iloc[0])))
    if rows:
        I3 = pd.DataFrame(rows)
        S = I3.groupby(["variant", "set", "intervention"]).agg(
            mean_gain_pp=("gain_pp", "mean"), sd=("gain_pp", "std"), seeds=("gain_pp", "size"),
            seeds_ci_above_0=("lo", lambda x: int((x > 0).sum())), seeds_ci_below_0=("hi", lambda x: int((x < 0).sum())))
        print("RQ3 interventions at the selected checkpoint (gain = learned - intervened, pp; > 0 = learned better). "
              "a1 = Mc only (Mc2 for no-spatial), a0 = Ms only, off = module removed (gamma = 0):")
        display(S.round(3)); S.to_csv(os.path.join(OUT, "tables", "rq3_interventions.csv"))
        for v in sorted(I3.variant.unique()):
            d = S.loc[v] if v in S.index.get_level_values(0) else None
            sd_img = G[G.Module.map(DISPLAY) == v]["rq3_val_gate_img_sd"].mean() if "rq3_val_gate_img_sd" in G else np.nan
            if d is not None and ("val", "mean") in d.index and ("val", "shuffle") in d.index:
                _n = int(d.loc[("val", "mean"), "seeds"]); _need = _n if _n < 5 else 4      # all seeds if < 5, else >= 4 / 5
                ok = (d.loc[("val", "mean"), "seeds_ci_above_0"] >= _need
                      and d.loc[("val", "shuffle"), "seeds_ci_above_0"] >= _need)
                print(f"  {v:18s} gate SD/img {sd_img:.4f} -> " + ("level 2 reached: per-image routing is FUNCTIONAL"
                      if ok else "level 2 NOT reached: the gate acts as a per-block constant (static recalibration)"))
        gc = [f"rq3_val_gate_{c}" for c in CLASS_NAMES if f"rq3_val_gate_{c}" in G]
        if gc:
            print("mean gate per true class (validation; a gate that routes by lesion type differs between classes):")
            display(G.groupby("Module")[gc].mean().rename(index=DISPLAY, columns=lambda c: c.replace("rq3_val_gate_", "")).round(4))
