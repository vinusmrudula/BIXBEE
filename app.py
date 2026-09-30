"""
BixBee | SIH26142 | Sentinel-2 4x super-resolution demo backend.

Run:   uvicorn app:app --port 8000
Open:  http://127.0.0.1:8000        (the backend serves index.html itself)

Everything the demo needs sits next to this file:
    sr_inference.py                              model + tiled MC-Dropout inference
    weights/ntro26142_swinir_x4_final.pth        trained checkpoint (~8.5 MB)
    samples/sample_01.npy, sample_01_hr.png ...  optional demo tiles (see README)
"""
import os
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from matplotlib import colormaps
from PIL import Image

from sr_inference import as_chw, load_model, super_resolve

BASE = Path(__file__).resolve().parent
UPLOADS, OUTPUTS, SAMPLES = BASE / "uploads", BASE / "outputs", BASE / "samples"
for folder in (UPLOADS, OUTPUTS, SAMPLES):
    folder.mkdir(exist_ok=True)

MODEL_NAME = "ntro26142_swinir_x4_final.pth"
MC_PASSES = int(os.environ.get("MC_PASSES", 5))
MAX_LR_SIDE = int(os.environ.get("MAX_LR_SIDE", 512))  # keeps CPU demo runs short
# Channel order of the training data is B, G, R, NIR (same guess the Colab notebook uses
# for its figures), so the RGB preview reads channels 2, 1, 0. Override with DISPLAY_BANDS="0,1,2".
DISPLAY_BANDS = [int(b) for b in os.environ.get("DISPLAY_BANDS", "2,1,0").split(",")]


def find_model_file() -> Path:
    candidates = [os.environ.get("MODEL_PATH"), BASE / "weights" / MODEL_NAME, BASE / MODEL_NAME]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise RuntimeError(
        f"Model file not found. Put {MODEL_NAME} in the 'weights' folder next to app.py "
        "(or set MODEL_PATH)."
    )


model, norm = load_model(str(find_model_file()))

app = FastAPI(title="BixBee Satellite Super Resolution")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
app.mount("/sample-files", StaticFiles(directory=SAMPLES), name="sample-files")


# ----------------------------------------------------------------------------- helpers
def rgb_preview(arr_chw: np.ndarray) -> Image.Image:
    """Colour preview. One stretch shared by all three bands so colours are not distorted."""
    if arr_chw.shape[0] < 3:
        arr_chw = np.repeat(arr_chw[:1], 3, axis=0)
        bands = arr_chw
    else:
        bands = arr_chw[DISPLAY_BANDS[:3]]
    bands = np.nan_to_num(bands.astype(np.float32))
    lo, hi = np.percentile(bands, [2, 98])
    rgb = np.clip((bands - lo) / (hi - lo + 1e-8), 0, 1)
    return Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))


def uncertainty_preview(unc: np.ndarray) -> Image.Image:
    scaled = np.clip(unc / (np.percentile(unc, 99) + 1e-8), 0, 1)
    rgba = colormaps["inferno"](scaled)
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))


def run_pipeline(lr_array: np.ndarray, label: str, reference_url=None) -> dict:
    try:
        lr_chw = as_chw(lr_array, norm["num_ch"]).astype(np.float32)
    except ValueError:
        raise HTTPException(
            400,
            f"Expected a {norm['num_ch']}-band array (H x W x {norm['num_ch']} or "
            f"{norm['num_ch']} x H x W), got shape {tuple(lr_array.shape)}.",
        )
    _, h, w = lr_chw.shape
    if max(h, w) > MAX_LR_SIDE:
        raise HTTPException(
            413, f"Tile is {h}x{w}. Please use tiles up to {MAX_LR_SIDE}x{MAX_LR_SIDE} pixels."
        )

    started = time.time()
    result = super_resolve(model, norm, lr_chw, mc_passes=MC_PASSES)
    seconds = time.time() - started

    sr, unc = result["mean"], result["uncertainty"]
    run_id = uuid.uuid4().hex[:8]
    rgb_preview(lr_chw).save(OUTPUTS / f"{run_id}_input.png")
    rgb_preview(sr).save(OUTPUTS / f"{run_id}_sr.png")
    uncertainty_preview(unc).save(OUTPUTS / f"{run_id}_uncertainty.png")
    np.save(OUTPUTS / f"{run_id}_sr.npy", sr)

    return {
        "message": "Super-resolution completed",
        "label": label,
        "run_id": run_id,
        "input_shape": list(lr_chw.shape),
        "output_shape": list(sr.shape),
        "seconds": round(seconds, 1),
        "mc_passes": MC_PASSES,
        "mean_uncertainty": round(float(unc.mean()), 4),
        "input_preview": f"/outputs/{run_id}_input.png",
        "sr_preview": f"/outputs/{run_id}_sr.png",
        "uncertainty_preview": f"/outputs/{run_id}_uncertainty.png",
        "sr_npy": f"/outputs/{run_id}_sr.npy",
        "reference_preview": reference_url,
    }


# ----------------------------------------------------------------------------- routes
@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "scale": norm["scale"], "bands": norm["num_ch"], "mc_passes": MC_PASSES}


@app.get("/samples")
def list_samples():
    items = []
    for f in sorted(SAMPLES.glob("*.npy")):
        has_ref = (SAMPLES / f"{f.stem}_hr.png").exists()
        items.append({"name": f.stem, "has_reference": has_ref})
    return {"samples": items}


@app.post("/enhance_sample")
def enhance_sample(name: str):
    path = SAMPLES / f"{Path(name).name}.npy"  # Path(...).name blocks path tricks
    if not path.exists():
        raise HTTPException(404, f"Sample '{name}' not found.")
    ref = None
    if (SAMPLES / f"{path.stem}_hr.png").exists():
        ref = f"/sample-files/{path.stem}_hr.png"
    return run_pipeline(np.load(path, allow_pickle=False), path.stem, ref)


@app.post("/enhance")
def enhance_upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".npy"):
        raise HTTPException(400, "Please upload a .npy file (4-band Sentinel-2 tile).")
    try:
        array = np.load(file.file, allow_pickle=False)
    except Exception:
        raise HTTPException(400, "Could not read that file as a NumPy .npy array.")
    return run_pipeline(array, file.filename)
