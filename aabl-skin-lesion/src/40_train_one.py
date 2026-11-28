# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 4.1  Train one configuration (verbose, with a guard against silently skipped updates)
#      + optional checkpoints: CKPT_EVERY (resume file) and SNAPSHOT_EPOCHS (kept fp16 weights), both off by default
# =============================================================================
VERBOSE   = True    # one line per epoch
LOG_EVERY = 1       # print every N epochs (validation epochs are always printed)


class OutOfTime(Exception):
    pass


class NonFiniteTraining(RuntimeError):
    pass


def metrics(P, y):
    pred = P.argmax(1)
    try:
        auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(NUM_CLASSES)))
    except ValueError:
        auc = np.nan
    return dict(BalancedAcc=balanced_accuracy_score(y, pred), Accuracy=accuracy_score(y, pred),
                MacroF1=f1_score(y, pred, average="macro", zero_division=0), MacroAUC=auc,
                QWK=cohen_kappa_score(y, pred, weights="quadratic"))      # proposal metric (class index read as ordinal)


@torch.no_grad()
def predict(model, loader):
    model.eval(); P = []
    for x, _ in loader:
        with autocast():
            P.append(torch.softmax(model(x).float(), 1).cpu())
    return torch.cat(P).numpy()


def run_key(table, backbone, module, pretrained, seed, fraction=1.0):
    """same key as the earlier run for 100 % data; data-fraction runs get _f25 / _f50"""
    f = "" if fraction == 1.0 else f"_f{int(round(100 * fraction))}"
    return f"{table}_{backbone}_{'in1k' if pretrained else 'scratch'}_{module}{f}_s{seed}"


def make_optimizer(model, lr0):
    """Protocol: one SGD group. Recipe fields (CFG.ATTN_LR_MULT, GAMMA_LR_MULT, GAMMA_WD) only split off the
    parameters that live inside attention modules ('.attn.'), for every module type alike."""
    gam = [p for n, p in model.named_parameters() if ".attn." in n and n.endswith(".gamma")]
    att = [p for n, p in model.named_parameters() if ".attn." in n and not n.endswith(".gamma")]
    ids = {id(p) for p in gam + att}
    rest = [p for p in model.parameters() if id(p) not in ids]
    groups = [dict(params=rest, lr=lr0, weight_decay=CFG.WEIGHT_DECAY, tag="backbone+fc"),
              dict(params=att, lr=lr0 * CFG.ATTN_LR_MULT, weight_decay=CFG.WEIGHT_DECAY, tag="attn"),
              dict(params=gam, lr=lr0 * CFG.GAMMA_LR_MULT, weight_decay=CFG.WEIGHT_DECAY if CFG.GAMMA_WD else 0.0,
                   tag="gamma")]
    return torch.optim.SGD([g for g in groups if g["params"]], lr=lr0, momentum=CFG.MOMENTUM)


def make_optimizers(model, lr0, module):
    """protocol SGD for backbone + fc; AAB-L parameters -> AdamW (module optimiser, AABL_CFG) when attn_adamw."""
    if module not in VARIANTS or not VARIANTS[module]["attn_adamw"]:
        return [make_optimizer(model, lr0)]
    grp = {"mask": [], "gate": [], "gain": []}; rest = []
    for n, p in model.named_parameters():
        if ".attn." not in n:
            rest.append(p)
        elif n.endswith(".raw_gamma"):
            grp["gain"].append(p)
        elif ".attn.gate" in n:
            grp["gate"].append(p)
        else:
            grp["mask"].append(p)
    sgd = torch.optim.SGD([dict(params=rest, lr=lr0, weight_decay=CFG.WEIGHT_DECAY, tag="backbone+fc")],
                          lr=lr0, momentum=CFG.MOMENTUM)
    adw = torch.optim.AdamW([dict(params=ps, lr=AABL_CFG["adamw_lr"][k], tag=f"{k}(adamw)",
                                  weight_decay=0.0 if k == "gain" else AABL_CFG["adamw_wd"])
                             for k, ps in grp.items() if ps])
    return [sgd, adw]


def train_one(table, backbone, module, pretrained, seed, fraction=1.0):
    key = run_key(table, backbone, module, pretrained, seed, fraction)
    model = build_model(backbone, module, pretrained, seed)
    lr0 = CFG.LR_PRETRAINED if pretrained else CFG.LR_SCRATCH
    opts = make_optimizers(model, lr0, module)
    aabs = []                                                    # (old AAB removed)
    aabls = [b for b in model.modules() if isinstance(b, AABL)]
    scheds = [torch.optim.lr_scheduler.StepLR(o, step_size=CFG.STEP_EPOCHS, gamma=0.1) for o in opts]
    _ids = [id(p) for o in opts for g in o.param_groups for p in g["params"]]
    assert len(_ids) == len(set(_ids)) == len([p for p in model.parameters() if p.requires_grad]), "optimizer groups invalid"
    w = None
    if CFG.USE_CLASS_WEIGHTS:
        cnt = np.bincount(y_all[TRAIN_FRACTION[fraction]], minlength=NUM_CLASSES)
        w = torch.tensor(cnt.sum() / (NUM_CLASSES * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    tr, va, _ = loaders(seed, fraction)
    yv = y_all[SPLIT["val"]]
    ck = os.path.join(OUT, "weights", key + ".ckpt")
    hist, start = [], 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEVICE, weights_only=False)
        if s.get("cfg") == CFG_HASH and s.get("impl", "") == impl_tag(module):
            model.load_state_dict(s["model"])
            for o, sd_ in zip(opts, s["opt"]):
                o.load_state_dict(sd_)
            for sc_, sd_ in zip(scheds, s["sched"]):
                sc_.load_state_dict(sd_)
            scaler.load_state_dict(s["scaler"]); hist, start = s["hist"], s["next"]
            log(f"    resume {key} at epoch {start + 1}")
        else:
            log(f"    stale checkpoint for {key} (config or AAB implementation changed) -> start over")
    if VERBOSE:
        log(f"    {backbone} + {module} | {'ImageNet' if pretrained else 'scratch'} | seed {seed} | "
              f"{n_params(model) / 1e6:.2f} M params | {CFG.EPOCHS} epochs | "
              + " | ".join(f"{g['tag']} lr {g['lr']:.0e} wd {g['weight_decay']:.0e}" for o in opts for g in o.param_groups))
    best = max([h.get("val_BalancedAcc", -1) for h in hist], default=-1)
    fc0 = model.fc.weight.detach().clone()
    tr.epoch = start
    t0 = time.time()
    for ep in range(start, CFG.EPOCHS):
        te = time.time(); model.train()
        tl = torch.zeros((), device=DEVICE); tc = torch.zeros((), device=DEVICE); nseen = 0
        lr_now = opts[0].param_groups[0]["lr"]                   # lr used in this epoch
        for b in aabs:                                           # optional gamma warm-up (frozen gamma)
            b.gamma.requires_grad_(ep >= CFG.GAMMA_WARMUP_EPOCHS and CFG.AAB_GAMMA_FIXED is None)
        rf = min(1.0, max(0.0, (ep - AABL_CFG["route_start"] + 1) / AABL_CFG["route_ramp"]))
        for b in aabls:                                          # AAB-L: per-image router term f(ep)
            b.res_factor = rf
        gstat = {"mean": [], "sd": [], "grad": []}
        for si, (x, y) in enumerate(tr):
            y = y.to(DEVICE, non_blocking=True)
            trk = bool(aabls) and si % 25 == 0
            for b in aabls:
                b.track = trk
            with autocast():
                out = model(x); loss = crit(out, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if trk:                                              # gate statistics + |dL/da| (unscaled), no effect on the step
                _s = float(scaler.get_scale()) if USE_AMP else 1.0
                gstat["mean"].append(float(torch.stack([b.last_a_pre.float().mean() for b in aabls]).mean()))
                gstat["sd"].append(float(torch.stack([b.last_a_pre.float().std(unbiased=False) for b in aabls]).mean()))
                _gg = [b.a_g.grad.abs().mean() / _s for b in aabls if b.a_g is not None and b.a_g.grad is not None]
                if _gg:
                    gstat["grad"].append(float(torch.stack(_gg).mean()))
                for b in aabls:
                    b.track = False; b.a_g = None
            for o in opts:
                scaler.step(o)
            scaler.update()
            tl += loss.detach().float() * len(y); tc += (out.detach().argmax(1) == y).sum(); nseen += len(y)
        for sc_ in scheds:
            sc_.step()
        row = dict(epoch=ep + 1, lr=lr_now, train_loss=float(tl) / nseen, train_acc=float(tc) / nseen,
                   sec=time.time() - te)
        if aabs:
            _g = torch.cat([b.gamma.detach().float().clamp(0, 1).flatten() for b in aabs])
            row.update(gamma_mean=_g.mean().item(), gamma_max=_g.max().item())
        if aabls:
            _g = torch.cat([b.gain().detach().float().flatten() for b in aabls])
            row.update(gamma_mean=_g.mean().item(), gamma_max=_g.max().item(), route_factor=rf,
                       gate_mean=float(np.mean(gstat["mean"])) if gstat["mean"] else np.nan,
                       gate_img_sd=float(np.mean(gstat["sd"])) if gstat["sd"] else np.nan,
                       gate_grad=float(np.mean(gstat["grad"])) if gstat["grad"] else np.nan)

        # guard: GradScaler silently skips every step when gradients are inf/NaN -> nothing is trained
        scale = scaler.get_scale() if USE_AMP else 1.0
        if (not math.isfinite(row["train_loss"]) or scale < 2.0 ** -4
                or (ep == start and torch.equal(model.fc.weight.detach(), fc0))):
            raise NonFiniteTraining(f"{key}: epoch {ep + 1} made no valid update (loss {row['train_loss']:.4f}, "
                                    f"grad scale {scale:g}) -> non-finite gradients; run is not recorded")

        is_val = (ep + 1) % CFG.VAL_EVERY == 0 or ep + 1 == CFG.EPOCHS
        if is_val:
            row.update({f"val_{k}": v for k, v in metrics(predict(model, va), yv).items()})
        hist.append(row)
        if VERBOSE and (is_val or (ep + 1) % LOG_EVERY == 0 or ep == start):
            done_ep = ep + 1 - start
            eta = (time.time() - t0) / done_ep * (CFG.EPOCHS - ep - 1) / 60
            msg = (f"    ep {ep + 1:3d}/{CFG.EPOCHS} | lr {lr_now:.0e} | loss {row['train_loss']:.4f} | "
                   f"tr-acc {row['train_acc']:.4f} | {row['sec']:.0f}s | eta {eta:.1f}m")
            if aabs or aabls:
                msg += f" | gamma {row['gamma_mean']:.4f} (max {row['gamma_max']:.4f})"
            if aabls:
                msg += (f" | gate {row['gate_mean']:.3f} SD/img {row['gate_img_sd']:.4f} |dL/da| {row['gate_grad']:.1e}"
                        f" f(ep) {rf:.2f}")
            if is_val:
                star = " *" if row["val_BalancedAcc"] > best else ""
                best = max(best, row["val_BalancedAcc"])
                msg += (f" || val bal {row['val_BalancedAcc']:.4f} acc {row['val_Accuracy']:.4f} "
                        f"F1 {row['val_MacroF1']:.4f} AUC {row['val_MacroAUC']:.4f}{star}")
            log(msg)
        # ---- optional checkpoints (section 14 turns them on; protocol runs are unchanged when off) ----
        _ce = globals().get("CKPT_EVERY", 0)
        if _ce and (ep + 1) % _ce == 0 and ep + 1 < CFG.EPOCHS:             # resume point, overwritten
            torch.save(dict(model=model.state_dict(), opt=[o.state_dict() for o in opts], sched=[q.state_dict() for q in scheds],
                            scaler=scaler.state_dict(), hist=hist, next=ep + 1, cfg=CFG_HASH,
                            impl=impl_tag(module)), ck)
        if (ep + 1) in globals().get("SNAPSHOT_EPOCHS", ()):                 # kept snapshot of the weights (fp16)
            torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                       os.path.join(OUT, "weights", f"{key}_ep{ep + 1}.pt"))
        if ep + 1 < CFG.EPOCHS and hours_used() + 2 * (time.time() - te) / 3600 > CFG.TIME_BUDGET_H:
            torch.save(dict(model=model.state_dict(), opt=[o.state_dict() for o in opts], sched=[q.state_dict() for q in scheds],
                            scaler=scaler.state_dict(), hist=hist, next=ep + 1, cfg=CFG_HASH,
                            impl=impl_tag(module)), ck)
            raise OutOfTime(f"{key}: stopped after epoch {ep + 1}, checkpoint saved")
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "history", key + ".csv"), index=False)
    if os.path.exists(ck):
        os.remove(ck)
    for b in aabls:
        b.res_factor = 1.0                                       # inference uses the full router term
    if VERBOSE and best >= 0:
        log(f"    best val bal-acc {best:.4f} (reporting uses the final epoch, as in the paper)")
    return model, (time.time() - t0) / 60


# ---- pre-flight: one AMP step per module must give finite gradients and move the head ----------
def preflight(backbone, module, pretrained):
    m = build_model(backbone, module, pretrained, 0); m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
    x, y = next(iter(loaders(0)[0]))
    with autocast():
        loss = F.cross_entropy(m(x), y.to(DEVICE))
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    fc0 = m.fc.weight.detach().clone(); opt.step()
    ok = math.isfinite(loss.item()) and not bad and not torch.equal(fc0, m.fc.weight.detach())
    del m, opt
    if USE_AMP:
        torch.cuda.empty_cache()
    return ok, bad[:3]


_bb = CFG.TABLE_II_BACKBONE if CFG.RUN_TABLE_II else CFG.TABLE_I_BACKBONES[0]      # only the modules trained here
_pre = CFG.RUN_TABLE_II and _bb in LOCAL_SD
for mod in [v for v in VARIANTS if VARIANTS[v]["tier"] in RUN_TIERS]:
    ok, bad = preflight(_bb, mod, _pre)
    print(f"pre-flight {_bb} + {mod:4s}: {'ok' if ok else 'NON-FINITE GRADIENTS ' + str(bad)}")
    assert ok, f"{mod}: gradients are not finite -> fix before running the grid"
