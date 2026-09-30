# BixBee | SIH26142 | Sentinel-2 4x Super-Resolution (SwinIR + MC-Dropout)

Smart India Hackathon 2026, Problem Statement SIH26142 (NTRO): Deep Learning Based Super Resolution Mapping from medium-resolution satellite imagery.

A SwinIR transformer that turns 10 m Sentinel-2 imagery (4 bands) into 2.5 m imagery (4x), with a per-pixel uncertainty map from 5-pass Monte Carlo Dropout.

**Demo video:** <add link>  |  **Drive folder (video, screenshots, notebook):** <add link>

## Results (held-out test regions, never used in training)

| Method | PSNR (dB) | SSIM | SAM (deg) |
|---|---|---|---|
| Bicubic | 12.20 | 0.3835 | 7.49 |
| SRCNN | 16.61 | 0.4522 | 4.99 |
| SwinIR, MC-Dropout mean of 5 (ours) | 17.16 | 0.4778 | 4.85 |

Sentinel-2 and the NAIP reference differ in sensor, acquisition date and spectral response, which caps absolute scores for every method. The comparison that matters is the gain over the baselines on identical data.

## Data and training

- 2,800 paired patches (700 regions x 4 patches): Sentinel-2 LR 64x64 (10 m), NAIP HR 256x256 (2.5 m), 4 bands (B, G, R, NIR).
- Split by region: train 540 ROIs / 2,160 patches, validation 60 ROIs / 240 patches, test 100 ROIs / 400 patches (never seen in training).
- SwinIR with 1.99 M parameters, 80 epochs on a single NVIDIA T4 (Colab), best checkpoint = epoch 31 (by validation PSNR).
- PSNR and SSIM are computed on normalized (0-1) values; SAM on original physical values.

## Method

- Paired Sentinel-2 (LR, 64x64) / NAIP (HR, 256x256) patches, 4 bands.
- Split by region (ROI), never by patch, to prevent spatial leakage. Separate LR and HR normalization.
- SwinIR predicts a residual on top of the bicubic image. Loss = Charbonnier + 0.5 x SAM (spectral angle).
- Inference is tiled (64 px tiles, 16 px overlap, feathered blending) so any tile size works.
- MC-Dropout (5 passes): mean = output image, std = uncertainty map. Fine detail is inferred by the model, and the uncertainty map shows where it is least reliable.

## Repo layout

```
app.py                      FastAPI backend (serves index.html too)
index.html                  demo UI (compare slider, side by side, uncertainty view)
sr_inference.py             model + tiled MC-Dropout inference + CLI
make_samples.py             builds demo tiles in samples/ from paired test patches
weights/ntro26142_swinir_x4_final.pth   trained checkpoint (~8.5 MB)
samples/                    demo tiles: sample_01.npy (+ sample_01_hr.png reference)
training/ntro_26142_swinir_colab.py     full Colab pipeline (data, baselines, training, test)
requirements.txt
```

## Run the demo

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
# open http://127.0.0.1:8000
```

The checkpoint file is a normal PyTorch `.pth` (it looks like a zip internally). Do not unzip it.

## Command line

```bash
python sr_inference.py --input lr.npy --model weights/ntro26142_swinir_x4_final.pth --out result --passes 5
```

Input: `.npy`, 4 bands, shape (C,H,W) or (H,W,C), in original Sentinel-2 units. Output: `result_sr.npy` (4 x 4H x 4W), `result_std.npy`, `result_uncertainty.npy` and a heatmap PNG.

## Make demo samples

```bash
python make_samples.py --pair path/to/lr_patch.npy,path/to/hr_patch.npy --name sample_01 --hr-order 0,1,2,3
```

The training notebook found the HR band order already matches (no reordering), so `--hr-order 0,1,2,3` is correct.

## Settings (environment variables)

`MC_PASSES` (default 5), `MAX_LR_SIDE` (default 512), `DISPLAY_BANDS` (default `2,1,0`, the RGB preview channels), `MODEL_PATH`.

## Reading the uncertainty map

The map shows where the 5 dropout passes disagree with each other (relative model disagreement). On our test set its correlation with the actual absolute error is weak (0.011), so treat it as a relative flag for regions to review, not a calibrated error estimate. Calibrating it is future work.

## Limits

Trained on paired Sentinel-2 / NAIP patches, so it is tuned to that domain. Input must be a 4-band `.npy` tile. Output detail is model-inferred, not observed.
