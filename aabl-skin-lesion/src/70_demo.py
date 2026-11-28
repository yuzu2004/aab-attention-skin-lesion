# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 7  Demonstration interface (proposal): upload a dermoscopy image -> class probabilities + Grad-CAM (AAB-L, seed 0)
#    Off by default. On Kaggle: pip install gradio, set AABL_DEMO=1, run; a public share link is printed.
#    Research prototype only — not a medical device.
# =============================================================================
DEMO = True
if DEMO:
    import gradio as gr
    _demo_model = load_kept("AABL", CFG.SEEDS[0])
    assert _demo_model is not None, "AAB-L seed-0 weights not found"

    def _infer(im):
        a = np.asarray(Image.fromarray(im).convert("RGB").resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR))
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(DEVICE).float() / 255
        xn = (x - on_cur(MEAN)) / on_cur(STD)
        with torch.no_grad():
            p = torch.softmax(_demo_model(xn).float(), 1)[0].cpu().numpy()
        cam, _ = grad_cam(_demo_model, xn)
        heat = (plt.get_cmap("jet")(cam[0])[..., :3] * 255).astype(np.uint8)
        over = (0.55 * a + 0.45 * heat).astype(np.uint8)
        return {CLASS_NAMES[i]: float(p[i]) for i in range(NUM_CLASSES)}, over

    gr.Interface(fn=_infer, inputs=gr.Image(type="numpy", label="dermoscopy image"),
                 outputs=[gr.Label(num_top_classes=3, label="prediction"), gr.Image(label="Grad-CAM (AAB-L)")],
                 title="AAB-L (ResNet-50) — HAM10000 / ISIC 2018 Task 3 (research prototype, not for diagnosis)"
                 ).launch(share=True)
