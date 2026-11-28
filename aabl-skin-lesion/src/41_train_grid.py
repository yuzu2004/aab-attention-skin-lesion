# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 4.2  The grid — the AAB-L family by tiers (RUN_TIERS), 5 seeds, resumable; tier order, seed by seed. Runs one job per GPU in parallel (2 x T4 -> 2 jobs).
#      Every run is still the paper's protocol (batch 32, same lr, same seed = same batches); only WHERE it runs
#      changes. AAB rows from an older AAB implementation (NaN-gradient std descriptor) are dropped and retrained.
# =============================================================================
import queue

JOBS = []                                                   # (table, backbone, module, pretrained, seed, fraction)
_BB = CFG.TABLE_II_BACKBONE
for tier in sorted(RUN_TIERS):
    if tier in (1, 2, 3):
        _ss = CFG.SEEDS if tier == 1 else OTHER_SEEDS                         # tiers 2-3: fewer seeds (exploratory)
        JOBS += [("II", _BB, m, True, s, 1.0) for s in _ss for m in VARIANTS if VARIANTS[m]["tier"] == tier]
    if tier == 4:                                           # boundary conditions: data fraction, then from scratch
        JOBS += [("II", _BB, m, True, s, f) for f in FRACTIONS for s in OTHER_SEEDS for m in BOUNDARY_MODULES]
        JOBS += [("II", _BB, m, False, s, 1.0) for s in OTHER_SEEDS for m in BOUNDARY_MODULES]

_need = sorted({bb for _, bb, _, pre, _, _ in JOBS if pre} - set(LOCAL_SD))
assert not _need, f"no local ImageNet weights for {_need}: see cell 2.4 (nothing is downloaded)"

if os.path.exists(RES_CSV):
    _r = read_results()
    if "Impl" not in _r.columns:
        _r["Impl"] = ""
    _r["Impl"] = _r["Impl"].fillna("").astype(str)
    stale = _r.Module.isin(list(VARIANTS)) & (_r.Impl != _r.Module.map(impl_tag))
    if stale.any():
        print(f"dropping {int(stale.sum())} AAB-L-family result(s) of an older implementation / config (retrained)")
        for k in _r.loc[stale, "Key"]:
            for f in (os.path.join(OUT, "weights", k + ".pt"), os.path.join(OUT, "history", k + ".csv"),
                      os.path.join(OUT, "preds", k + ".npz")):
                if os.path.exists(f):
                    os.remove(f)
        _r = _r[~stale]
    _r.to_csv(RES_CSV, index=False)
    done = set(_r[_r.CfgHash == CFG_HASH].Key)
else:
    done = set()
todo = [j for j in JOBS if run_key(*j) not in done]
N_WORKERS = max(1, min(NGPU, len(todo)))
print(f"{len(JOBS)} runs planned, {len(JOBS) - len(todo)} on disk, {len(todo)} to train on {N_WORKERS} GPU(s)")

_CSV_LOCK = threading.Lock()
_STOP = threading.Event()
_Q = queue.Queue()
for j in todo:
    _Q.put(j)


def run_job(table, bb, mod, pre, seed, frac=1.0):
    key = run_key(table, bb, mod, pre, seed, frac)
    log(f"[{key}] start")
    model, mins = train_one(table, bb, mod, pre, seed, frac)
    _, _, te = loaders(seed)
    P = predict(model, te); yt = y_all[SPLIT["test"]]
    np.savez_compressed(os.path.join(OUT, "preds", key + ".npz"), y_true=yt, y_proba=P)
    row = dict(Key=key, Table=table, Backbone=bb, Init="ImageNet-1k" if pre else "scratch", Fraction=frac,
               Module=mod, Seed=seed, Params=n_params(model), TrainMin=round(mins, 1),
               SecPerEpoch=round(60 * mins / CFG.EPOCHS, 2), TrainSteps=len(loaders(seed, frac)[0]) * CFG.EPOCHS,
               CfgHash=CFG_HASH, Impl=impl_tag(mod), **metrics(P, yt))
    if GX_T is not None:
        Pt = predict(model, Loader(GX_T, GY_T, np.arange(len(GY_T)), 0, 128, False))
        row.update({f"isic_{k}": v for k, v in metrics(Pt, TEST_LABELS.y.values).items()})
        np.savez_compressed(os.path.join(OUT, "preds", key + "_isic.npz"), y_true=TEST_LABELS.y.values, y_proba=Pt)
    extra = ""
    if mod in VARIANTS:
        _g = torch.cat([b.gain().detach().flatten() for b in model.modules() if isinstance(b, AABL) and b.gain() is not None])
        extra = f" | gamma -> {_g.mean():.4f} [{_g.min():.4f}, {_g.max():.4f}]"
        if pre and frac == 1.0:                            # kept (fp16) for RQ2 / RQ3 / Grad-CAM / demo
            torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                       os.path.join(OUT, "weights", key + ".pt"))
    with _CSV_LOCK:
        pd.concat([read_results() if os.path.exists(RES_CSV) else pd.DataFrame(), pd.DataFrame([row])],
                  ignore_index=True).to_csv(RES_CSV, index=False)
    log(f"[{key}] DONE test bal {row['BalancedAcc']:.4f} | ISIC bal {row.get('isic_BalancedAcc', np.nan):.4f} | "
        f"{mins:.1f} min{extra}")
    del model
    if USE_AMP:
        torch.cuda.empty_cache()


def worker(dev):
    if USE_AMP:
        torch.cuda.set_device(dev)                         # thread-local: "cuda" now means cuda:dev here
        assert torch.empty(1, device=DEVICE).device.index == dev
    while not _STOP.is_set():
        try:
            job = _Q.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(*job)
        except OutOfTime as e:
            log("  " + str(e) + " -> re-run this cell in a new session to continue"); _STOP.set()
        except NonFiniteTraining as e:
            log("  !! " + str(e))
        except Exception as e:                             # keep the other GPU going, report the failure
            import traceback
            log(f"  !! {run_key(*job)} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")


t_grid = time.time()
threads = [threading.Thread(target=worker, args=(d,), name=f"gpu{d}") for d in range(N_WORKERS)]
for th in threads:
    th.start()
for th in threads:
    th.join()
print(f"\ngrid finished in {(time.time() - t_grid) / 3600:.2f} h; "
      f"{len(set(read_results().Key) & {run_key(*j) for j in JOBS}) if os.path.exists(RES_CSV) else 0}"
      f"/{len(JOBS)} runs on disk")
