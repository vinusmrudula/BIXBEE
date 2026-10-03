[README (2).md](https://github.com/user-attachments/files/33009199/README.2.md)
# TerraLens | SIH26142 | Sentinel-2 4x Super-Resolution (SwinIR + MC-Dropout)

Team BixBee, Smart India Hackathon 2026. Problem Statement SIH26142 (NTRO): Deep Learning Based Super Resolution Mapping from medium-resolution satellite imagery.

TerraLens uses a SwinIR transformer to turn 10 m Sentinel-2 imagery (4 bands) into 2.5 m imagery (4x sharper), and shows an uncertainty map from 5-pass Monte Carlo Dropout.

**Demo video:** https://drive.google.com/file/d/10jJdUDgy26aPMjJGIqp5SY9H3XVK1cTq/view?usp=drive_link
**Drive folder (video, screenshots, notebook):** https://drive.google.com/file/d/10jJdUDgy26aPMjJGIqp5SY9H3XVK1cTq/view?usp=drive_link

## Results (held-out test regions, never used in training)

| Method | PSNR (dB) | SSIM | SAM (deg) |
|---|---|---|---|
| Bicubic | 12.20 | 0.3835 | 7.49 |
| SRCNN | 16.61 | 0.4522 | 4.99 |
| SwinIR, MC-Dropout mean of 5 (ours) | 17.16 | 0.4778 | 4.85 |

PSNR/SSIM: higher is better. SAM (spectral angle): lower is better.

Sentinel-2 and the NAIP reference come from different sensors and dates, which caps absolute scores for every method. The comparison that matters is the gain over the baselines on identical data.

## Data and training

- 2,800 paired patches (700 regions x 4 patches): Sentinel-2 low-res 64x64 (10 m) and NAIP high-res 256x256 (2.5 m), 4 bands (B, G, R, NIR).
- Split by region, never by patch, to prevent spatial leakage: train 540 regions / 2,160 patches, validation 60 / 240, test 100 / 400 (never seen in training).
- Separate normalization for low-res and high-res images.
- SwinIR with 1.99 M parameters, 80 epochs on one NVIDIA T4 (Colab). Best checkpoint = epoch 31 (by validation PSNR).
- SwinIR predicts a correction on top of the bicubic image. Loss = Charbonnier + 0.5 x SAM (spectral angle).
- PSNR and SSIM are computed on normalized (0-1) values; SAM on original physical values.
- Full training pipeline: `NTRO_26142_SwinIR_fixed.ipynb`.

## How inference works

- Tiled (64 px tiles, 16 px overlap, feathered blending), so any tile size works.
- MC-Dropout (5 passes): the mean is the output image, the standard deviation is the uncertainty map.

## Files in this repo

```
app.py                              FastAPI backend (also serves index.html)
index.html                          demo UI (compare slider, side by side, uncertainty view)
sr_inference.py                     model + tiled MC-Dropout inference + command line
ntro26142_swinir_x4_final.pth       trained checkpoint (~8.6 MB)
NTRO_26142_SwinIR_fixed.ipynb       training and evaluation notebook (Colab)
samples/                            demo tiles: sample_0X.npy + sample_0X_hr.png (NAIP reference)
requirements.txt
```

The checkpoint is a normal PyTorch `.pth` file (it looks like a zip inside). Do not unzip it.

## Run the demo

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
# open http://127.0.0.1:8000
```

`app.py` finds the checkpoint in the main folder (or in `weights/`, or via the `MODEL_PATH` setting). Demo scenes are read from the `samples/` folder.

## Command line

```bash
python sr_inference.py --input lr.npy --model ntro26142_swinir_x4_final.pth --out result --passes 5
```

Input: `.npy`, 4 bands, shape (C,H,W) or (H,W,C), in original Sentinel-2 units. Output: `result_sr.npy` (4 x 4H x 4W), `result_std.npy`, `result_uncertainty.npy` and a heatmap PNG.

## Settings (environment variables)

`MC_PASSES` (default 5), `MAX_LR_SIDE` (default 512), `DISPLAY_BANDS` (default `2,1,0`, the RGB preview channels), `MODEL_PATH`.

## Reading the uncertainty map

The map shows where the 5 dropout passes disagree with each other (relative model disagreement). On our test set its link to the actual error is weak (correlation 0.011), so treat it as a pointer to regions worth a second look, not a calibrated error estimate. Calibrating it is future work.

## Limits

- Trained on paired Sentinel-2 / NAIP patches, so it is tuned to that kind of data.
- Input must be a 4-band `.npy` tile.
- Some fine detail in the output is the model's best estimate, not something the satellite observed.
