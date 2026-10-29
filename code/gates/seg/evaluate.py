"""Evaluate network labels against ACDC expert ED labels; write per-case metrics, G1 verdict and MANIFEST.tsv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seglib  # noqa: E402

NETS = list(seglib.LABEL_MAPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--seg", required=True, help="data/processed/acdc_seg")
    ap.add_argument("--metrics", required=True, help="output TSV of metrics")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--gate-pids", default="101,102,104")
    a = ap.parse_args()
    seg = Path(a.seg)
    rows = []
    for pid in seglib.pid_list(a.pids):
        inf = seglib.info(pid)
        img, gtp = seglib.ed_paths(pid)
        gt_img = nib.load(str(gtp))
        gt = np.asarray(gt_img.dataobj).astype(np.uint8)
        sp = gt_img.header.get_zooms()[:3]
        for net in NETS:
            p = seg / net / f"{pid}_ED.nii.gz"
            r = {"patient": pid, "group": inf["Group"], "ed_frame": int(inf["ED"]), "network": net,
                 "path": str(p.relative_to(seg.parent.parent.parent)) if p.exists() else "", "status": "ok"}
            if not p.exists():
                r["status"] = "missing"
                rows.append(r)
                continue
            pim = nib.load(str(p))
            pred = np.asarray(pim.dataobj).astype(np.uint8)
            if pred.shape != gt.shape or not np.allclose(pim.affine, gt_img.affine, atol=1e-4):
                r["status"] = "geometry_mismatch"
            r.update(seglib.evaluate(pred, gt, sp))
            rows.append(r)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("patient", "group", "ed_frame",
                                                                          "network", "path", "status"), k))
    with open(a.metrics, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + "\n")
    gate = seglib.pid_list(a.gate_pids)
    verdict = {}
    for net in NETS:
        rs = [r for r in rows if r["network"] == net and r["patient"] in gate]
        verdict[net] = bool(len(rs) == len(gate) and all(r.get("usable_all") for r in rs))
    n_ok = sum(verdict.values())
    out = {"gate_pids": gate, "usable_by_network": verdict, "n_usable": n_ok, "G1_pass": n_ok >= 3}
    print(json.dumps(out, indent=1))
    Path(a.metrics).with_suffix(".gate.json").write_text(json.dumps(out, indent=1))
    if a.manifest:
        with open(a.manifest, "w") as f:
            f.write("patient\tgroup\tnetwork\tpath\tdice_LV\tdice_MYO\tdice_RV\tdice_raw_LV\tdice_raw_MYO\t"
                    "dice_raw_RV\thd95_LV_mm\thd95_MYO_mm\thd95_RV_mm\tusable_all\n")
            for r in rows:
                if r["status"] == "missing":
                    continue
                v = [r["patient"], r["group"], r["network"], r["path"]] + \
                    [f"{r[k]:.4f}" for k in ["LV_dice", "MYO_dice", "RV_dice", "LV_dice_raw", "MYO_dice_raw",
                                             "RV_dice_raw", "LV_hd95", "MYO_hd95", "RV_hd95"]] + [str(r["usable_all"])]
                f.write("\t".join(v) + "\n")


if __name__ == "__main__":
    main()
