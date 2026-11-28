# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 6  Grad-CAM (last stage, predicted class) for NA (earlier run) and AAB-L (seed 0, final epoch), 2 ISIC images per class
# =============================================================================
def load_kept(module, seed):
    """AAB-L family: this run; NA: the earlier run's kept weights (attach its output folder)"""
    if module in VARIANTS:
        return load_trained(module, seed)
    p = _find_file(run_key("II", CFG.TABLE_II_BACKBONE, module, True, seed) + ".pt", "weights")
    if p is None:
        return None
    m = build_model(CFG.TABLE_II_BACKBONE, module, False, seed)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in torch.load(p, map_location=DEVICE).items()})
    return m.eval()


def grad_cam(model, x):
    """x: [B,3,H,W] normalised -> cam [B,H,W] in [0,1], predicted classes"""
    feats = {}
    h = model.layer4.register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
    try:
        x = x.float().requires_grad_(False)
        out = model(x); pred = out.argmax(1)
        a = feats["a"]; g = torch.autograd.grad(out.gather(1, pred[:, None]).sum(), a)[0]
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    finally:
        h.remove()
    return cam.detach().cpu().numpy(), pred.cpu().numpy()


_models = {DISPLAY[v]: m for v in ("NA", "AABL") for m in [load_kept(v, CFG.SEEDS[0])] if m is not None}
if "AAB-L" in _models and GX_T is not None:
    yT = TEST_LABELS.y.values; rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.flatnonzero(yT == c), min(2, (yT == c).sum()), replace=False)
                           for c in range(NUM_CLASSES) if (yT == c).any()])
    xs = ((on_cur(GX_T)[torch.as_tensor(pick, device=on_cur(GX_T).device)].float() / 255 - on_cur(MEAN)) / on_cur(STD))
    cams = {v: grad_cam(m, xs) for v, m in _models.items()}
    img = (on_cur(GX_T)[torch.as_tensor(pick, device=on_cur(GX_T).device)].permute(0, 2, 3, 1).cpu().numpy())
    fig, ax = plt.subplots(1 + len(_models), len(pick), figsize=(1.6 * len(pick), 1.8 * (1 + len(_models))))
    for i in range(len(pick)):
        ax[0, i].imshow(img[i]); ax[0, i].set_title(CLASS_NAMES[yT[pick[i]]], fontsize=8)
        for r, v in enumerate(_models, start=1):
            ax[r, i].imshow(img[i]); ax[r, i].imshow(cams[v][0][i], cmap="jet", alpha=0.45)
            ax[r, i].set_title(f"{v}: {CLASS_NAMES[cams[v][1][i]]}", fontsize=7)
    for a_ in ax.ravel():
        a_.axis("off")
    plt.tight_layout(); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True); plt.savefig(os.path.join(OUT, "figures", "gradcam_na_vs_aabl.png"), dpi=150); plt.show()
else:
    print("Grad-CAM needs the AAB-L seed-0 weights (and, for the NA row, the earlier run attached) and the ISIC test set")
