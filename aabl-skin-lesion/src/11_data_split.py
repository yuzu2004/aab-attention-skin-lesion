# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 1.2  Lesion-grouped split 5/1/1 (the paper's own split is not described)
# =============================================================================
y_all = LABELS.y.values
GROUPED = cover > 0.9
if GROUPED:
    folds = [te for _, te in StratifiedGroupKFold(CFG.N_SPLIT_FOLDS, shuffle=True, random_state=CFG.SPLIT_SEED)
             .split(np.zeros(len(y_all)), y_all, LABELS.lesion.values)]
else:
    folds = [te for _, te in StratifiedKFold(CFG.N_SPLIT_FOLDS, shuffle=True, random_state=CFG.SPLIT_SEED)
             .split(np.zeros(len(y_all)), y_all)]
SPLIT = {"test": np.sort(folds[0]), "val": np.sort(folds[1]), "train": np.sort(np.concatenate(folds[2:]))}
if GROUPED:
    gs = {k: set(LABELS.lesion.values[v]) for k, v in SPLIT.items()}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"])
    print("group audit: OK, zero lesion overlap across train / val / test")
else:
    print("!! no lesion ids: image-level split (leaky, optimistic)")
display(pd.DataFrame({k: np.bincount(y_all[v], minlength=NUM_CLASSES) for k, v in SPLIT.items()},
                     index=CLASS_NAMES).T)
