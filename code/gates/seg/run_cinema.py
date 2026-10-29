"""CineMA acdc_sax segmentation on ACDC ED frames WITHOUT the ground-truth-centred crop.

CineMA's ACDC preprocessing (cinema/data/acdc/preprocess.py) crops 192 x 192 in-plane around the bounding box of the
EXPERT LV label. That uses ground truth at test time, so it is replaced here by an image-derived crop:

  pass 1: the resampled (1 x 1 x 10 mm) ED volume is padded to >= 192 x 192 x 16 and segmented with MONAI sliding-
          window inference (roi 192 x 192 x 16, overlap 0.5, gaussian blending) over the whole field of view;
  pass 2: the largest 3D connected component of the pass-1 LV cavity prediction gives a bounding box, and CineMA's own
          get_center_crop_size_from_bbox is applied to that PREDICTED box (the training-time crop rule, with the
          prediction in place of the expert label). The crop is segmented again.
  If pass 1 finds no LV, the crop is centred on the image centre (logged).

The three released seeds (acdc_sax_0/1/2) are ensembled by averaging softmax probabilities. Probabilities are
resampled back to the original image grid (linear) before argmax, so output labels share the original affine.
Intensity handling copies the CineMA preprocessing: clip to the 0.95 and 99.5 percentiles, sitk Normalize, rescale
to [0, 1], cast to uint8 (x 255), then ScaleIntensity to [0, 1] as in the inference example.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seglib  # noqa: E402
from cinema.data.sitk import (  # noqa: E402
    cast_to_uint8,
    clip_and_normalise_intensity_3d,
    get_center_crop_size_from_bbox,
    resample_spacing_3d,
)
from cinema.segmentation.convunetr import get_model  # noqa: E402

WDIR = Path("/project/home/p201509/work/cvri/simulation/data/raw/weights/cinema_acdc_sax/finetuned/segmentation/acdc_sax")
SPACING = (1.0, 1.0, 10.0)
ROI = (192, 192, 16)


def load_models(device):
    cfg = OmegaConf.load(WDIR / "config.yaml")
    models = []
    for s in range(3):
        sd = {}
        with safe_open(str(WDIR / f"acdc_sax_{s}.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        m = get_model(cfg)
        m.load_state_dict(sd)
        m.eval().to(device)
        models.append(m)
    return models


def predict_probs(models, vol: np.ndarray, device) -> np.ndarray:
    """vol (x, y, z) in [0, 1]; returns softmax probs (4, x, y, z) averaged over seeds, sliding window."""
    x, y, z = vol.shape
    px, py, pz = max(ROI[0], x), max(ROI[1], y), max(ROI[2], z)
    pad = np.zeros((px, py, pz), np.float32)
    pad[:x, :y, :z] = vol  # pad at the end, as SpatialPadd(method="end") in CineMA
    t = torch.from_numpy(pad)[None, None].to(device)
    acc = None
    for m in models:
        def f(inp, m=m):
            return m({"sax": inp})["sax"]
        with torch.no_grad():
            logits = sliding_window_inference(t, ROI, 1, f, overlap=0.5, mode="gaussian")
        p = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        acc = p if acc is None else acc + p
    return (acc / len(models))[:, :x, :y, :z]


def run_one(models, pid: str, out_root: Path, device) -> dict:
    img_path, _ = seglib.ed_paths(pid)
    orig = sitk.ReadImage(str(img_path))
    res = resample_spacing_3d(image=orig, is_label=False, target_spacing=SPACING)
    res_n = cast_to_uint8(clip_and_normalise_intensity_3d(res, intensity_range=None))
    vol = np.transpose(sitk.GetArrayFromImage(res_n)).astype(np.float32)  # (x, y, z)
    vol = (vol - vol.min()) / max(vol.max() - vol.min(), 1e-8)  # ScaleIntensity
    # pass 1: whole field of view
    p1 = predict_probs(models, vol, device)
    lv1, _ = seglib.largest_cc(np.argmax(p1, 0) == 3)
    fallback = not lv1.any()
    if fallback:
        bmin = np.array([vol.shape[0] // 2, vol.shape[1] // 2, 0])
        bmax = bmin + 1
        bmax[2] = vol.shape[2]
    else:
        idx = np.nonzero(lv1)
        bmin = np.array([i.min() for i in idx])
        bmax = np.array([i.max() + 1 for i in idx])
    lo, up = get_center_crop_size_from_bbox(bbox_min=bmin, bbox_max=bmax, current_size=vol.shape,
                                            target_size=(*ROI[:2], vol.shape[2]))
    sl = tuple(slice(lo[i], vol.shape[i] - up[i]) for i in range(3))
    # pass 2: predicted-LV-centred crop
    p2 = predict_probs(models, vol[sl], device)
    probs = np.zeros_like(p1)
    probs[0] = 1.0  # background outside the crop
    probs[(slice(None),) + sl] = p2
    # back to original grid: each class probability resampled linearly, then argmax
    chans = []
    for c in range(4):
        im = sitk.GetImageFromArray(np.transpose(probs[c]).astype(np.float32))
        im.CopyInformation(res)
        chans.append(sitk.GetArrayFromImage(sitk.Resample(im, orig, sitk.Transform(), sitk.sitkLinear, 0.0)))
    lab_zyx = np.argmax(np.stack(chans), 0).astype(np.uint8)
    lab = np.transpose(lab_zyx)  # (x, y, z) matches nibabel array order
    lab = seglib.remap(lab, "cinema_acdc_sax")
    out = out_root / "cinema_acdc_sax" / f"{pid}_ED.nii.gz"
    seglib.save_like(lab, img_path, out)
    return {"pid": pid, "crop_lower": [int(v) for v in lo], "crop_upper": [int(v) for v in up],
            "fallback_centre_crop": bool(fallback)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    models = load_models(device)
    log = [run_one(models, p, Path(a.out), device) for p in seglib.pid_list(a.pids)]
    print(json.dumps({"device": str(device), "seconds": time.time() - t0, "cases": log}))
    (Path(a.out) / "cinema_acdc_sax" / "crop_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
