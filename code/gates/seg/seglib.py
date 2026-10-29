"""Shared helpers for the G1 segmentation gate (seg owner, I11 gates run, 2026-10-28).

Label convention for every saved output: ACDC expert convention
    0 background, 1 RV cavity, 2 LV myocardium, 3 LV cavity.
"""
from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ACDC_ROOT = Path("/project/home/p201509/work/cvri/simulation/data/raw/acdc/extracted")
STRUCT = {"RV": 1, "MYO": 2, "LV": 3}  # ACDC convention

# network-native label -> ACDC label; verified against source on 2026-10-28:
#   Task027: nnunet/dataset_conversion/Task027_AutomaticCardiacDetectionChallenge.py {1 RV, 2 MLV, 3 LVC}
#   Task114: nnunet/dataset_conversion/Task114_heart_MNMs.py {1 LVBP, 2 LVM, 3 RV}
#   UKB SAX (Dataset100): dataset.json {1 LV blood pool, 2 LV myocardium, 3 RV blood pool}
#   CineMA: cinema/__init__.py RV_LABEL 1, MYO_LABEL 2, LV_LABEL 3
#   ukbb_cardiac: common/cardiac_utils.py label = {'LV': 1, 'Myo': 2, 'RV': 3}
LABEL_MAPS = {
    "nnunet_task027_acdc": {0: 0, 1: 1, 2: 2, 3: 3},
    "nnunet_task114_mnms": {0: 0, 1: 3, 2: 2, 3: 1},
    "nnunet_ukb_sax": {0: 0, 1: 3, 2: 2, 3: 1},
    "cinema_acdc_sax": {0: 0, 1: 1, 2: 2, 3: 3},
    "ukbb_cardiac_fcn_sa": {0: 0, 1: 3, 2: 2, 3: 1},
}


def info(pid: str) -> dict:
    split = "testing" if int(pid[-3:]) > 100 else "training"
    d = {}
    for line in (ACDC_ROOT / split / pid / "Info.cfg").read_text().splitlines():
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    d["split"] = split
    d["dir"] = ACDC_ROOT / split / pid
    return d


def ed_paths(pid: str) -> tuple[Path, Path]:
    d = info(pid)
    ed = int(d["ED"])
    return d["dir"] / f"{pid}_frame{ed:02d}.nii.gz", d["dir"] / f"{pid}_frame{ed:02d}_gt.nii.gz"


def remap(arr: np.ndarray, net: str) -> np.ndarray:
    out = np.zeros(arr.shape, np.uint8)
    for src, dst in LABEL_MAPS[net].items():
        out[arr == src] = dst
    return out


def largest_cc(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest 3D 26-connected component and the fraction of voxels it keeps."""
    n = int(mask.sum())
    if n == 0:
        return mask.astype(bool), 0.0
    lab, k = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    if k == 1:
        return mask.astype(bool), 1.0
    sizes = ndimage.sum(mask, lab, index=np.arange(1, k + 1))
    keep = int(np.argmax(sizes)) + 1
    return lab == keep, float(sizes[keep - 1] / n)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def hd95(a: np.ndarray, b: np.ndarray, spacing) -> float:
    """Symmetric 95th percentile Hausdorff distance (mm) between mask surfaces."""
    a = a.astype(bool)
    b = b.astype(bool)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    sa, sb = _surface(a), _surface(b)
    dta = ndimage.distance_transform_edt(~sa, sampling=spacing)
    dtb = ndimage.distance_transform_edt(~sb, sampling=spacing)
    d = np.concatenate([dtb[sa], dta[sb]])
    return float(np.percentile(d, 95))


def evaluate(pred: np.ndarray, gt: np.ndarray, spacing) -> dict:
    """Metrics per structure for raw and largest-CC masks; gate fields per pre-declared definition."""
    row = {}
    for name, lab in STRUCT.items():
        p = pred == lab
        g = gt == lab
        lcc, frac = largest_cc(p)
        row[f"{name}_present"] = bool(p.any())
        row[f"{name}_lcc_frac"] = frac
        row[f"{name}_dice_raw"] = dice(p, g)
        row[f"{name}_hd95_raw"] = hd95(p, g, spacing)
        row[f"{name}_dice"] = dice(lcc, g)
        row[f"{name}_hd95"] = hd95(lcc, g, spacing)
        row[f"{name}_usable"] = bool(p.any() and row[f"{name}_dice"] > 0.70)
    row["usable_all"] = all(row[f"{n}_usable"] for n in STRUCT)
    return row


def save_like(arr: np.ndarray, ref_path: Path, out_path: Path) -> None:
    """Save uint8 labels with the original affine and header of ref_path."""
    ref = nib.load(str(ref_path))
    img = nib.Nifti1Image(arr.astype(np.uint8), ref.affine, ref.header)
    img.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


def pid_list(spec: str) -> list[str]:
    out = []
    for tok in spec.split(","):
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            out += [f"patient{i:03d}" for i in range(int(m[1]), int(m[2]) + 1)]
        else:
            out.append(f"patient{int(tok):03d}")
    return out
