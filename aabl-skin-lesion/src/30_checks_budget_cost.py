# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 3.1  Unit checks (identity at gamma = 0, fast form == explicit formula), 23-25 M budget, capacity-matched
#      controls, measured size / speed of every module (no training)
# =============================================================================
with torch.no_grad():
    for c in (64, 256, 2048):
        x = torch.randn(2, c, 14, 14)
        for v, cls in MODULE_FACTORY.items():
            if cls is not None:
                assert cls(c).eval()(x).shape == x.shape, (v, c)
        for v in VARIANTS:
            blk = MODULE_FACTORY[v](c).eval(); blk.gamma_override = 0.0
            assert torch.equal(blk(x), x), f"{v}: not exactly the identity at gamma = 0"
        blk = MODULE_FACTORY["AABL_NAIVE"](c).eval()
        assert torch.equal(blk(x), x), "naive block (gamma0 = 0) is not the identity at initialisation"
print("shapes ok | every symmetric / suppress block == identity at gamma = 0 (exact) | naive block == identity at init")


def aab_selftest(variant, c=256, hw=14):
    """fast factorised forward == explicit M formula, values and gradients (fp32, eval mode, router switched on)"""
    torch.manual_seed(0); m = MODULE_FACTORY[variant](c).eval(); m.res_factor = 1.0
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        if m.raw_gamma is not None and m.cfg["gamma_param"] != "sigmoid":
            m.raw_gamma.clamp_(0.05, 0.95)
        m.res_run.fill_(0.05)
    x = torch.randn(4, c, hw, hw, requires_grad=True)
    ps = [p for p in m.parameters()]
    y1 = m(x); g1 = torch.autograd.grad(y1.square().sum(), [x] + ps, allow_unused=True)
    y0 = m.forward_reference(x); g0 = torch.autograd.grad(y0.square().sum(), [x] + ps, allow_unused=True)
    err = (y1 - y0).abs().max().item()
    gerr = max(((p - q).abs().max() / q.abs().max().clamp_min(1e-12)).item() for p, q in zip(g1, g0) if q is not None)
    return dict(variant=variant, max_abs_diff=err, max_rel_grad_diff=gerr, gate_sd_4img=float(m.last_a.std()))


_st = pd.DataFrame([aab_selftest(v) for v in VARIANTS])
display(_st.round(8))
assert (_st.max_abs_diff < 1e-4).all() and (_st.max_rel_grad_diff < 1e-3).all(), "AAB-L self-test failed"

# ---- parameters; capacity-matched controls must have >= the parameters of the full block (proposal) -----------------
CAP = pd.DataFrame([dict(variant=v, tier=VARIANTS[v]["tier"] if v in VARIANTS else "ref",
                         params=n_params(build_model(CFG.TABLE_II_BACKBONE, v, False, 0)))
                    for v in ("NA", "SE", "CBAM", "ECA", "TA") + tuple(VARIANTS)]).set_index("variant")
CAP["params_M"] = CAP.params / 1e6
CAP["overhead_vs_NA_%"] = 100 * (CAP.params / CAP.loc["NA", "params"] - 1)
display(CAP.round(3))
CAPACITY_OK = all(CAP.loc[v, "params"] >= CAP.loc["AABL", "params"] for v in ABLATION_CONTROLS)
print(f"capacity-matched controls (>= AAB-L = {CAP.loc['AABL', 'params_M']:.3f} M): {'OK' if CAPACITY_OK else 'VIOLATED'}")
assert CAPACITY_OK, "a capacity-matched control has fewer parameters than AAB-L"
_lo, _hi = AABL_CFG["param_budget"]
assert _lo <= CAP.loc["AABL", "params"] <= _hi, "AAB-L outside the 23-25 M parameter budget"
print(f"AAB-L {CAP.loc['AABL', 'params_M']:.2f} M parameters: inside the {_lo / 1e6:.0f}-{_hi / 1e6:.0f} M budget")


def model_cost_table(variants, bb, b_train=32, b_inf=128, reps=10):
    """size and MEASURED speed of the whole network (same GPU, AMP as in training). GFLOPs count conv / matmul only."""
    out = []
    for v in variants:
        m = build_model(bb, v, False, 0)
        r = dict(variant=DISPLAY.get(v, v), ParamsM=n_params(m) / 1e6)
        m.eval(); x1 = torch.zeros(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=DEVICE).contiguous(memory_format=MEMFMT)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x1)
        r["GFLOPs"] = fc.get_total_flops() / 1e9
        if USE_AMP:
            m.train(); x = torch.randn(b_train, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=DEVICE).contiguous(memory_format=MEMFMT)
            for i in range(reps + 3):
                if i == 3:
                    torch.cuda.synchronize(); t0 = time.time()
                with autocast():
                    loss = m(x).float().sum()
                loss.backward(); m.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); r["train_ms_b32"] = (time.time() - t0) / reps * 1000
            m.eval(); xi = torch.randn(b_inf, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=DEVICE).contiguous(memory_format=MEMFMT)
            with torch.no_grad():
                for i in range(reps + 3):
                    if i == 3:
                        torch.cuda.synchronize(); t0 = time.time()
                    with autocast():
                        m(xi)
            torch.cuda.synchronize(); r["infer_img_s"] = b_inf / ((time.time() - t0) / reps)
            del x, xi
        out.append(r); del m
        if USE_AMP:
            torch.cuda.empty_cache()
    d = pd.DataFrame(out).set_index("variant")
    for c_ in ("GFLOPs", "train_ms_b32"):
        if c_ in d and "NA" in d.index:
            d[f"{c_}_vs_NA_%"] = 100 * (d[c_] / d.loc["NA", c_] - 1)
    return d


COST = model_cost_table(["NA", "SE", "CBAM", "ECA", "TA", "AABL"], CFG.TABLE_II_BACKBONE)
COST.to_csv(os.path.join(OUT, "tables", "cost.csv"))
print("size and measured speed (AMP; train = fwd+bwd batch 32; inference = batch 128):")
display(COST.round(3))
