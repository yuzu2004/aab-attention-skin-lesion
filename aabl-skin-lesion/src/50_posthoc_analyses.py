# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 5.0  Per-run analyses of the trained AAB-L family (post hoc, from the kept weights; cached in tables/posthoc.csv)
#      RQ2 gate-gradient probe | RQ3 gate interventions | block activity ||F' - F|| / ||F||  (dose-response)
# =============================================================================
def _bal(y, p):
    cm = np.bincount(y * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES); n = cm.sum(1)
    return 100 * float(np.mean(np.diag(cm)[n > 0] / n[n > 0]))


def gate_grad_probe(model, n_img=512, bs=64):
    """RQ2: per block gamma, |Mc - Ms|, per-image |dl_i/da_i| (CE, fp32, eval-mode BN), gate mean / SD over images"""
    blocks = gate_blocks(model)
    if not blocks:
        return {}
    model.eval()
    for b in blocks:
        b.probe = True
    acc = [{"g": [], "dm": [], "grad": [], "a": []} for _ in blocks]
    try:
        for x, y in Loader(GX, GY, SPLIT["val"][:n_img], 0, bs, False):
            x = x.float(); y = y.to(x.device)
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x).float(), y, reduction="sum").backward()
            for k, b in enumerate(blocks):
                po = b.probe_out; a = po["a"]
                acc[k]["grad"].append((a.grad.detach().abs().flatten() if a.grad is not None else torch.zeros(a.numel())).cpu())
                acc[k]["dm"].append((po["mc"].detach()[:, :, None, None] - po["ms"].detach()).abs().mean((1, 2, 3)).cpu())
                acc[k]["a"].append(a.detach().flatten().cpu())
                acc[k]["g"].append(float(b.gain().mean()) if b.gain() is not None else np.nan)
    finally:
        for b in blocks:
            b.probe = False; b.probe_out = None
        model.zero_grad(set_to_none=True)
    return dict(probe_gamma=float(np.nanmean([np.mean(v["g"]) for v in acc])),
                probe_mc_minus_ms=float(np.mean([torch.cat(v["dm"]).mean() for v in acc])),
                probe_grad_gate=float(np.mean([torch.cat(v["grad"]).mean() for v in acc])),
                probe_gate_img_sd=float(np.mean([torch.cat(v["a"]).std() for v in acc])))


@torch.no_grad()
def run_gates(model, gx, idx, given=None, const=None, gamma0=False, bs=128, measure=False):
    """probabilities [N,7] and gates [N,K]; given = [N,K] per-image gates, const = [K] per-block gates"""
    model.eval(); blocks = gate_blocks(model); idx = np.asarray(idx); P, A = [], []
    for b in blocks:
        b.measure = measure; b.act = []
    try:
        for a0 in range(0, len(idx), bs):
            sl = slice(a0, a0 + bs)
            gxc = on_cur(gx); j = torch.as_tensor(idx[sl], dtype=torch.long, device=gxc.device)
            x = ((gxc[j].float().div_(255) - on_cur(MEAN)) / on_cur(STD)).contiguous(memory_format=MEMFMT)
            for k, b in enumerate(blocks):
                b.a_given = None if given is None else torch.as_tensor(given[sl, k], dtype=torch.float32)
                b.a_const = None if const is None else float(const[k])
                b.gamma_override = 0.0 if gamma0 else None
            with autocast():
                P.append(torch.softmax(model(x).float(), 1).cpu())
            if blocks:
                A.append(torch.stack([b.last_a.float().cpu() for b in blocks], 1))
        act = (np.mean([torch.cat(b.act).mean().item() for b in blocks]) if measure and blocks else np.nan)
    finally:
        for b in blocks:
            b.a_given = b.a_const = b.gamma_override = None; b.measure = False; b.act = []
    return torch.cat(P).numpy(), (torch.cat(A).numpy() if A else None), act


def intervention_audit(model, gx, idx, y, groups=None, seed=0, n_boot=1000):
    """RQ3 at the selected checkpoint: gain = bal(learned) - bal(intervened), pp, > 0 = learned better; bootstrap 95% CI
    over lesions (HAM) or images (ISIC). mean / shuffle test per-image routing; a1 = Mc only (Mc2 for no-spatial),
    a0 = Ms only; off = gamma 0 (module removed)."""
    rng = np.random.default_rng(100 + seed); y = np.asarray(y); mem = None
    if groups is not None:
        _, gi = np.unique(np.asarray(groups), return_inverse=True); mem = [np.flatnonzero(gi == j) for j in range(gi.max() + 1)]
    P0, A, act = run_gates(model, gx, idx, measure=True); p0 = P0.argmax(1)
    out = dict(gate_mean=float(A.mean()), gate_img_sd=float(A.std(0).mean()), activity=float(act))
    K = A.shape[1]
    for name, kw in (("mean", dict(const=A.mean(0))), ("shuffle", dict(given=A[rng.permutation(len(A))])),
                     ("a1", dict(const=np.ones(K))), ("a0", dict(const=np.zeros(K))), ("off", dict(gamma0=True))):
        pi = run_gates(model, gx, idx, **kw)[0].argmax(1); bs = []
        for _ in range(n_boot):
            ix = (rng.integers(0, len(y), len(y)) if mem is None
                  else np.concatenate([mem[j] for j in rng.integers(0, len(mem), len(mem))]))
            bs.append(_bal(y[ix], p0[ix]) - _bal(y[ix], pi[ix]))
        out[f"{name}_gain_pp"] = _bal(y, p0) - _bal(y, pi)
        out[f"{name}_lo"], out[f"{name}_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    for c_, nm in enumerate(CLASS_NAMES):
        if (y == c_).any():
            out[f"gate_{nm}"] = float(A[y == c_].mean())
    return out


def gamma_stats(model):
    g = [b.gain() for b in gate_blocks(model) if b.gain() is not None]
    if not g:
        return {}
    g = torch.cat([x.detach().float().flatten() for x in g])
    return dict(gamma_mean=g.mean().item(), gamma_min=g.min().item(), gamma_max=g.max().item())


def _find_file(name, sub):
    hits = [os.path.join(OUT, sub, name)] + sorted(glob.glob(f"/kaggle/input/**/{sub}/{name}", recursive=True))
    return next((h for h in hits if os.path.exists(h)), None)


def load_trained(module, seed):
    p = _find_file(run_key("II", CFG.TABLE_II_BACKBONE, module, True, seed) + ".pt", "weights")
    if p is None:
        return None
    m = build_model(CFG.TABLE_II_BACKBONE, module, False, seed)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in torch.load(p, map_location=DEVICE).items()})
    for b in gate_blocks(m):
        b.res_factor = 1.0
    return m.eval()


POSTHOC_CSV = os.path.join(OUT, "tables", "posthoc.csv")
PH = pd.read_csv(POSTHOC_CSV) if os.path.exists(POSTHOC_CSV) else pd.DataFrame(columns=["Key"])
_sets = [("val", GX, SPLIT["val"], y_all[SPLIT["val"]], LABELS.lesion.values[SPLIT["val"]] if "lesion" in LABELS else None)]
if GX_T is not None:
    _sets.append(("isic", GX_T, np.arange(len(GY_T)), TEST_LABELS.y.values, None))
for mod in VARIANTS:
    for seed in CFG.SEEDS:
        key = run_key("II", CFG.TABLE_II_BACKBONE, mod, True, seed)
        if key in set(PH.Key):
            continue
        m = load_trained(mod, seed)
        if m is None:
            continue
        row = dict(Key=key, Module=mod, Seed=seed, **gamma_stats(m), **gate_grad_probe(m))
        for nm, gx, idx, yy, grp in _sets:
            row.update({f"rq3_{nm}_{k}": v for k, v in intervention_audit(m, gx, idx, yy, grp, seed).items()})
        PH = pd.concat([PH, pd.DataFrame([row])], ignore_index=True); PH.to_csv(POSTHOC_CSV, index=False)
        log(f"post-hoc analyses {DISPLAY[mod]} s{seed} done"); del m
print(f"post-hoc analyses available for {len(PH)} trained AAB-L-family models")
