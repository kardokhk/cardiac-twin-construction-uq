"""Known-answer tests for seglib metrics and label maps. Run: python -I test_seglib.py (exit 0 = pass)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seglib  # noqa: E402


def test_dice():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True  # 64 voxels
    b = np.zeros_like(a); b[3:7, 2:6, 2:6] = True  # overlap 48
    assert abs(seglib.dice(a, a) - 1.0) < 1e-12
    assert abs(seglib.dice(a, b) - 2 * 48 / 128) < 1e-12
    assert np.isnan(seglib.dice(np.zeros_like(a), np.zeros_like(a)))


def test_hd95():
    a = np.zeros((30, 30, 30), bool); a[5:15, 5:15, 5:15] = True
    assert seglib.hd95(a, a, (1, 1, 1)) == 0.0
    b = np.roll(a, 3, axis=0)  # shift 3 voxels along x
    h = seglib.hd95(a, b, (2.0, 1, 1))  # 3 voxels x 2 mm = 6 mm upper bound
    assert 0 < h <= 6.0 + 1e-9, h
    assert np.isnan(seglib.hd95(a, np.zeros_like(a), (1, 1, 1)))


def test_lcc():
    m = np.zeros((10, 10, 10), bool); m[0:3, 0:3, 0:3] = True; m[6:8, 6:8, 6:8] = True  # 27 + 8
    k, f = seglib.largest_cc(m)
    assert k.sum() == 27 and abs(f - 27 / 35) < 1e-12
    d = np.zeros((5, 5, 5), bool); d[1, 1, 1] = True; d[2, 2, 2] = True  # diagonal: 26-connected
    assert seglib.largest_cc(d)[1] == 1.0


def test_remap():
    x = np.array([0, 1, 2, 3], np.uint8)
    assert seglib.remap(x, "nnunet_task027_acdc").tolist() == [0, 1, 2, 3]
    for net in ("nnunet_task114_mnms", "nnunet_ukb_sax", "ukbb_cardiac_fcn_sa"):
        assert seglib.remap(x, net).tolist() == [0, 3, 2, 1]  # LV<->RV swap
    assert seglib.remap(x, "cinema_acdc_sax").tolist() == [0, 1, 2, 3]


def test_expert_self():
    import nibabel as nib
    _, gtp = seglib.ed_paths("patient101")
    g = nib.load(str(gtp))
    gt = np.asarray(g.dataobj).astype(np.uint8)
    r = seglib.evaluate(gt, gt, g.header.get_zooms()[:3])
    for s in ("LV", "MYO", "RV"):
        assert r[f"{s}_dice"] == 1.0 and r[f"{s}_hd95"] == 0.0, (s, r)
    assert r["usable_all"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
