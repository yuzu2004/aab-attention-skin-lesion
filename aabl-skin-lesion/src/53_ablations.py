# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.3  Capacity-matched ablations: AAB-L minus each control (one component removed, control >= AAB-L parameters)
#      positive = the removed component helps.
# =============================================================================
if SHOW and "AABL" in set(NEW_MAIN.Module):
    rows = []
    for c, comp in zip(ABLATION_CONTROLS, ("amplification (symmetric vs suppress-only)", "adaptive gating (vs learned constant)",
                                           "learned gate (vs fixed 0.5)", "spatial descriptor (vs second channel mask)")):
        for col in (PRIMARY, "BalancedAcc", "isic_MacroF1"):
            r = paired("AABL", c, col, data=NEW_MAIN)
            if r:
                rows.append(dict(component=comp, params_control_M=CAP.loc[c, "params_M"], params_AABL_M=CAP.loc["AABL", "params_M"], **r))
    if rows:
        T2 = holm(pd.DataFrame(rows)); T2["verdict"] = [verdict(r) for _, r in T2.iterrows()]
        print("Table 2 — capacity-matched ablations (pp; same run and seeds; Holm over the 4 components per metric):")
        display(T2.round(4)); T2.to_csv(os.path.join(OUT, "tables", "table2_ablation.csv"), index=False)
    else:
        print("tier-2 runs not finished yet")
