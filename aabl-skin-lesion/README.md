# AAB-L — Optimizing Symmetric Residual Attention Networks: An Optimization-Aware Training Perspective

Capstone project (FPT University, AIP491). AAB-L is a light symmetric residual attention block for ResNet-50 that can
both **amplify and suppress** features, with a learned per-image gate:

```
M  = a·Mc + (1 − a)·Ms          Mc: channel mask, Ms: spatial mask, a: per-image gate (router)
F′ = F · (1 + γ(2M − 1))        γ = sigmoid(raw_gamma), learned per channel
```

Blocks are placed in stages 3–4 of an ImageNet-pretrained ResNet-50 (24.14 M parameters in total, budget 23–25 M).
Training follows the protocol of Pedro & Oliveira (2022, Table II) on HAM10000 with a lesion-grouped 5/1/1 split,
evaluated on the internal test split and on the official ISIC 2018 Task 3 test set.

## Repository layout

```
src/
  run.py                            runs the step files in order, in one shared namespace
  00_runtime.py                     imports, devices, AMP, logging
  01_config.py                      protocol configuration (CFG, AABL_CFG, tiers, seeds)
  05_download_data.py               automatic download of the data and ImageNet weights (kagglehub)
  10_data_locate.py                 find HAM10000 / ISIC 2018 test data and labels
  11_data_split.py                  lesion-grouped 5/1/1 split + zero-overlap audit
  12_data_decode.py                 decode once at 224x224 (uint8 on GPU)
  20_reference_modules.py           SE, CBAM, ECA, Triplet Attention (references)
  21_aabl_module.py                 the proposed AAB-L block and its family (controls, lever variants)
  22_resnet_integration.py          ResNet with a module after every block
  23_imagenet_weights.py            local ImageNet weights matched by content
  30_checks_budget_cost.py          unit checks, parameter budget, capacity matching, measured cost
  40_train_one.py                   train one configuration
  41_train_grid.py                  tiered, resumable training grid
  50_posthoc_analyses.py            per-run post hoc analyses
  51_integrity_guards.py            integrity guards, seed-noise floor
  52_rq1_comparison.py              RQ1: AAB-L vs NA, SE, CBAM, ECA, TA
  53_ablations.py                   capacity-matched ablations
  54_rq2_optimisation.py            RQ2: identity-at-init saddle, anti-saddle levers
  55_rq3_gate.py                    RQ3: static recalibration vs per-image routing
  56_rq4_boundary.py                RQ4: data size, pretrained vs scratch
  60_gradcam.py                     Grad-CAM
  70_demo.py                        Gradio demo (only with AABL_DEMO=1)
requirements.txt
```

Pipeline steps: 0 runtime and configuration · 1 data (location, lesion-grouped split, decoding) · 2 modules
(reference SE / CBAM / ECA / Triplet Attention, AAB-L, ResNet integration, ImageNet weights) · 3 unit checks, parameter
budget, measured cost · 4 training (one run; tiered, resumable grid) · 5 results (integrity guards, RQ1 comparison,
capacity-matched ablations, RQ2 optimisation / saddle, RQ3 gate behaviour, RQ4 boundary conditions) · 6 Grad-CAM ·
7 demonstration interface.

## Research questions

- **RQ1** Does AAB-L beat the plain backbone (NA) and SE, CBAM, ECA, TA?
- **RQ2** Why does an identity-at-init block not learn (the saddle at γ = 0), and which lever of the recipe fixes it?
- **RQ3** Does the gate route per image, or act as a per-block constant?
- **RQ4** Under which data-size and pre-training conditions does the block help?

## Quick start

Everything is automatic: the data (HAM10000 + labels, Kaggle dataset
[`yuw1409/data-isic2018`](https://www.kaggle.com/datasets/yuw1409/data-isic2018)) and the ImageNet ResNet-50 weights
(Kaggle model `yuw1409/model-resnet`) are downloaded by `src/05_download_data.py` with `kagglehub` on the first run,
then the whole pipeline runs from training to the result tables and figures.

**Kaggle notebook** (recommended: GPU T4 x2, Internet on)

```python
!git clone https://github.com/<user>/aabl-skin-lesion.git
%cd aabl-skin-lesion
!pip install -q -r requirements.txt
!python src/run.py
```

**Google Colab / local machine** (NVIDIA GPU recommended)

```bash
git clone https://github.com/<user>/aabl-skin-lesion.git
cd aabl-skin-lesion
pip install -r requirements.txt          # install the PyTorch build for your CUDA first
python src/run.py
```

Check the setup before the long training: `python src/run.py --until 30` (download, split, model checks, no training).

If Kaggle asks for credentials, create an API token (kaggle.com > Settings > API) and set `KAGGLE_USERNAME` and
`KAGGLE_KEY` (or place `kaggle.json` in `~/.kaggle/`).

## How to run

The step files share globals (each uses names defined by the earlier ones), so run them through `src/run.py`:

```bash
python src/run.py               # everything
python src/run.py --until 30    # stop before training
python src/run.py --skip 41     # skip the training grid, recompute results from weights already in the output folder
python src/run.py --list        # list the steps
```

- **Data and weights.** Step 05 downloads them only if they are not already present: attached Kaggle inputs under
  `/kaggle/input`, `CFG.DATA_ROOT` in `src/01_config.py`, or `./data_isic2018` and `./weights` on a local machine.
  Environment variables: `AABL_AUTO_DOWNLOAD=0` disables the download, `AABL_DATA_HANDLE` / `AABL_WEIGHTS_HANDLE` change
  the Kaggle source. ImageNet weights are always matched to their architecture by content (`src/23_imagenet_weights.py`).
- **External test.** The official ISIC 2018 Task 3 test set is evaluated when its images and ground-truth CSV are
  present in the data; otherwise that evaluation is skipped.
- **Configuration.** Main switches are in `src/01_config.py` (`CFG`, `AABL_CFG`, `RUN_TIERS`, `OTHER_SEEDS`). The full
  grid (all tiers, 5 seeds) needs tens of GPU hours; remove tiers from `RUN_TIERS` or seeds to shorten it.
- **Outputs.** `/kaggle/working/po_aab/` on Kaggle, `./po_aab/` elsewhere (`weights/`, `preds/`, `history/`, `tables/`,
  `figures/`, `results.csv`).
- **Resuming.** Finished runs found in the output folder (or attached under `/kaggle/input`) are restored and skipped, so
  an interrupted session continues where it stopped.
- **Demo.** The Gradio demo starts only with `AABL_DEMO=1`.

Reference modules (NA, SE, CBAM, ECA, TA) are **not retrained** here; their per-seed results from an earlier run with the
same protocol, split, seeds and batch order are built into `src/51_integrity_guards.py` (attach that run's `results.csv`
for every metric).

## Reading the results

A difference is reported as real only if the paired 95 % CI over seeds excludes 0 **and** its magnitude exceeds the
measured seed-noise floor; otherwise it is a tie. Ties and negative results are valid answers.

## Notes

- Model weights, caches and predictions are not stored in git (see `.gitignore`); publish them as a Kaggle dataset or
  a GitHub release if needed.
- The data are downloaded from the Kaggle dataset above; HAM10000 and ISIC 2018 remain subject to their original licences.
- Research prototype only — not a medical device.
