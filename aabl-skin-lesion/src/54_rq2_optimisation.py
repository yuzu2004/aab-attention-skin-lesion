# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.4  RQ2 — Optimisation: the identity-at-init saddle and the one-lever-at-a-time anti-saddle recipe
#   s = 1 + gamma (2M - 1)  ->  dL/dtheta_attn = sum dL/ds * gamma * 2 dM/dtheta_attn     (theta_attn = masks, gate)
#   => at gamma = 0 every mask / gate parameter has EXACTLY zero gradient (only gamma itself moves): a saddle that
#      "identity-at-init" designs start in. The gate gradient also scales with (Mc - Ms): symmetric init (Mc = Ms)
#      gives the gate no signal even when gamma > 0. With a straight-through clamp gamma itself still has a gradient
#      (escape route). With a HARD clamp (torch.clamp) the gradient at the boundary is version-dependent: on some PyTorch
#      versions it is 0 (absorbing: the block stays the identity), on others it passes — measured below, not assumed.
# =============================================================================
import matplotlib.pyplot as plt


def init_grad_groups(variant, n_img=64):
    """gradient norm per parameter group at initialisation (fp32, eval-mode BN, one validation batch, CE)"""
    m = build_model(CFG.TABLE_II_BACKBONE, variant, CFG.TABLE_II_BACKBONE in LOCAL_SD, 0).eval()   # ImageNet backbone
    for b in gate_blocks(m):
        b.res_factor = 1.0
    x, y = next(iter(Loader(GX, GY, SPLIT["val"][:n_img], 0, n_img, False)))
    F.cross_entropy(m(x.float()).float(), y.to(x.device)).backward()
    grp = {"mask": 0.0, "gate": 0.0, "gamma": 0.0}
    for n, p in m.named_parameters():
        if ".attn." not in n or p.grad is None:
            continue
        k = "gamma" if n.endswith("raw_gamma") else ("gate" if ".attn.gate" in n else "mask")
        grp[k] += float(p.grad.float().pow(2).sum())
    del m
    return {f"grad_norm_{k}": np.sqrt(v) for k, v in grp.items()}


G0 = pd.DataFrame([dict(variant=DISPLAY[v], **init_grad_groups(v)) for v in ("AABL_HARD",) + LEVER_VARIANTS]).set_index("variant")
print("Gradient norms at initialisation (the saddle: mask and gate gradients are exactly 0 when gamma0 = 0):")
display(G0)
G0.to_csv(os.path.join(OUT, "tables", "rq2_init_gradients.csv"))
for v in ("AAB-L-naive", "AAB-L-naive+init", "AAB-L-naive+lr"):
    assert G0.loc[v, "grad_norm_mask"] == 0 and G0.loc[v, "grad_norm_gate"] == 0 and G0.loc[v, "grad_norm_gamma"] > 0, \
        f"{v}: saddle check failed (gamma0 = 0 must give zero mask / gate gradient, non-zero gamma gradient)"
print("saddle check passed: gamma0 = 0 -> zero mask / gate gradient, non-zero gamma gradient (escape only through gamma)")
_hg = G0.loc["AAB-L-naive-hardclamp", "grad_norm_gamma"]
print(f"hard clamp (torch {torch.__version__}): gamma gradient at gamma = 0 = {_hg:.3e} -> "
      + ("ABSORBING on this PyTorch version (gamma can never leave 0)" if _hg == 0 else
         "NOT absorbing on this PyTorch version (torch.clamp passes the gradient at the boundary); whether a hard "
         "clamp traps gamma = 0 depends on the PyTorch version, so do not state it as a general result"))

if SHOW:
    rows = []
    for v in LEVER_VARIANTS[1:]:
        for col in (PRIMARY, "BalancedAcc"):
            r = paired(v, "AABL_NAIVE", col, data=NEW_MAIN)
            if r:
                rows.append(r)
    if rows:
        T3 = holm(pd.DataFrame(rows))
        T3["verdict"] = [verdict(r) for _, r in T3.iterrows()]
        print("Table 3 — one lever at a time, each vs the identity-at-init (naive) block (pp). Note: the naive / one-lever "
              "variants use gamma = clamp(raw) (straight-through) so that gamma0 = 0 exists; AAB-L uses sigmoid(raw):")
        display(T3.round(4)); T3.to_csv(os.path.join(OUT, "tables", "table3_levers.csv"), index=False)
    G = PH.merge(NEW_MAIN[["Key"] + [c for c in (PRIMARY, "BalancedAcc") if c in NEW_MAIN]], on="Key", how="left")
    pc = [c for c in ("gamma_mean", "probe_mc_minus_ms", "probe_grad_gate", "probe_gate_img_sd", "rq3_val_gate_img_sd") if c in G]
    if pc and len(G):
        print("Trained blocks (final epoch, mean over seeds): gamma, |Mc - Ms|, per-image |dl/da|, gate SD over images")
        display(G.groupby("Module")[pc].mean().rename(index=DISPLAY).round(5))

    # training dynamics of the lever variants (seed 0): gamma, gate SD over images, |dL/da|
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    for v in LEVER_VARIANTS:
        p = _find_file(run_key("II", CFG.TABLE_II_BACKBONE, v, True, CFG.SEEDS[0]) + ".csv", "history")
        if p:
            h = pd.read_csv(p)
            for a_, c_ in zip(ax, ("gamma_mean", "gate_img_sd", "gate_grad")):
                if c_ in h:
                    a_.plot(h.epoch, h[c_], label=DISPLAY[v])
    for a_, t_ in zip(ax, ("gamma (mean)", "gate SD over images", "|dL/da| (unscaled)")):
        a_.set_title(t_); a_.set_xlabel("epoch")
    ax[2].set_yscale("symlog", linthresh=1e-6); ax[0].legend(fontsize=7); plt.tight_layout()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True); plt.savefig(os.path.join(OUT, "figures", "rq2_dynamics.png"), dpi=150); plt.show()

    # activity vs performance (dose-response) across every AAB-family run
    if "rq3_val_activity" in G and len(G):
        Gd = G.dropna(subset=["rq3_val_activity"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for a_, col in zip(ax, (PRIMARY, "BalancedAcc")):
            if col in Gd and Gd[col].notna().sum() > 2:
                for v, d in Gd.groupby("Module"):
                    a_.scatter(d.rq3_val_activity, d[col], label=DISPLAY[v], s=18)
                rho, p = sps.spearmanr(Gd.rq3_val_activity, Gd[col], nan_policy="omit")
                a_.set_title(f"{col} vs activity (Spearman rho {rho:.2f}, p {p:.3f})"); a_.set_xlabel("||F' - F|| / ||F||")
        ax[0].legend(fontsize=6); plt.tight_layout()
        os.makedirs(os.path.join(OUT, "figures"), exist_ok=True); plt.savefig(os.path.join(OUT, "figures", "rq2_dose_response.png"), dpi=150); plt.show()
