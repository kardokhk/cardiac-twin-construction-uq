"""nnU-Net inference on ACDC ED frames for the three nnU-Net weight sets, then label remap to ACDC convention.

- nnunet_task027_acdc (nnU-Net v1.7.1, Zenodo 4485926): ensemble of 2d and 3d_fullres (5 folds each, nnUNetTrainerV2,
  nnUNetPlansv2.1), with the ensemble's own postprocessing.json; the shipped ensembles/summary.csv ranks this
  ensemble first (mean Dice 0.9228 in cross-validation).
- nnunet_task114_mnms (nnU-Net v1.7.1, Zenodo 4288464): ensemble of 2d and 3d_fullres (5 folds each,
  nnUNetTrainerV2_MMS), as in the M&Ms 2020 entry; no ensemble postprocessing file is shipped, so none is applied.
- nnunet_ukb_sax (nnU-Net v2.6.2, Zenodo 15649643, Dataset100_UKBB_Petersen_SAX): 2d, 5 folds, checkpoint_final.
Default nnU-Net test-time mirroring is kept for all three.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seglib  # noqa: E402

HERE = Path(__file__).resolve().parent
W = Path("/project/home/p201509/work/cvri/simulation/data/raw/weights")
BIN = Path(sys.executable).parent


def v1_results_tree(root: Path) -> Path:
    """RESULTS_FOLDER/nnUNet/<config>/<Task> symlinks onto the read-only extracted weights."""
    base = root / "nnUNet"
    links = {
        ("2d", "Task027_ACDC"): W / "nnunet_task027_acdc/extracted/2d/Task027_ACDC",
        ("3d_fullres", "Task027_ACDC"): W / "nnunet_task027_acdc/extracted/3d_fullres/Task027_ACDC",
        ("ensembles", "Task027_ACDC"): W / "nnunet_task027_acdc/extracted/ensembles/Task027_ACDC",
        ("2d", "Task114_heart_MNMs"): W / "nnunet_task114_mnms/extracted/2d/Task114_heart_MNMs",
        ("3d_fullres", "Task114_heart_MNMs"): W / "nnunet_task114_mnms/extracted/3d_fullres/Task114_heart_MNMs",
    }
    # nnU-Net v1 writes a training_log file into each fold dir at inference, and the raw weights are read-only,
    # so directories are recreated (writable) and only the files are symlinked.
    for (cfg, task), tgt in links.items():
        for src in tgt.rglob("*"):
            if src.is_file():
                dst = base / cfg / task / src.relative_to(tgt)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to(src)
    return root


def sh(cmd, env):
    print("+", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    subprocess.run(list(map(str, cmd)), check=True, env=env)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True, help="data/processed/acdc_seg")
    ap.add_argument("--work", required=True, help="job work dir for nnU-Net intermediate files")
    ap.add_argument("--nets", default="nnunet_task027_acdc,nnunet_task114_mnms,nnunet_ukb_sax")
    a = ap.parse_args()
    work = Path(a.work)
    inp = work / "input"
    inp.mkdir(parents=True, exist_ok=True)
    pids = seglib.pid_list(a.pids)
    for pid in pids:
        img, _ = seglib.ed_paths(pid)
        dst = inp / f"{pid}_0000.nii.gz"
        if not dst.exists():
            dst.symlink_to(img)
    env = dict(os.environ)
    env["RESULTS_FOLDER"] = str(v1_results_tree(work / "v1_results"))
    env["nnUNet_raw_data_base"] = str(work / "v1_raw")
    env["nnUNet_preprocessed"] = str(work / "preprocessed")
    env["nnUNet_raw"] = str(work / "v2_raw")
    env["nnUNet_results"] = str(W / "nnunet_ukb_sax/extracted/nnUNet_results")
    timing = {}
    nets = a.nets.split(",")
    v1 = {"nnunet_task027_acdc": ("Task027_ACDC", "nnUNetTrainerV2"),
          "nnunet_task114_mnms": ("Task114_heart_MNMs", "nnUNetTrainerV2_MMS")}
    # one prediction per GPU, launched concurrently (4 A100 per node; the UKB 2d model shares GPU 0)
    jobs = []
    for net in nets:
        if net in v1:
            task, tr = v1[net]
            for cfg in ["2d", "3d_fullres"]:
                jobs.append((net, cfg, [BIN / "nnUNet_predict", "-i", inp, "-o", work / net / cfg, "-t", task,
                                        "-m", cfg, "-tr", tr, "-p", "nnUNetPlansv2.1", "-z",
                                        "--num_threads_preprocessing", 6, "--num_threads_nifti_save", 6]))
        elif net == "nnunet_ukb_sax":
            jobs.append((net, "2d", [BIN / "nnUNetv2_predict", "-i", inp, "-o", work / net / "2d", "-d",
                                     "Dataset100_UKBB_Petersen_SAX", "-c", "2d", "-tr", "nnUNetTrainer",
                                     "-p", "nnUNetPlans", "-f", 0, 1, 2, 3, 4, "-npp", 6, "-nps", 6]))
    procs = []
    t0 = time.time()
    for i, (net, cfg, cmd) in enumerate(jobs):
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = str(i % 4)
        logf = open(work / f"{net}_{cfg}.log", "w")
        print("+", " ".join(map(str, cmd)), "GPU", i % 4, flush=True)
        procs.append((net, cfg, subprocess.Popen(list(map(str, cmd)), env=e, stdout=logf, stderr=subprocess.STDOUT),
                      logf))
    for net, cfg, pr, logf in procs:
        rc = pr.wait()
        logf.close()
        timing[f"{net}/{cfg}"] = time.time() - t0
        if rc != 0:
            print((work / f"{net}_{cfg}.log").read_text()[-3000:])
            raise SystemExit(f"{net} {cfg} failed rc={rc}")
    for net in nets:
        if net not in v1:
            continue
        cmd = [BIN / "nnUNet_ensemble", "-f", work / net / "2d", work / net / "3d_fullres", "-o", work / net / "ensemble"]
        if net == "nnunet_task027_acdc":
            cmd += ["-pp", W / "nnunet_task027_acdc/extracted/ensembles/Task027_ACDC/"
                    "ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/"
                    "postprocessing.json"]
        timing[f"{net}/ensemble"] = sh(cmd, env)
    # remap and save with original affine
    src_dir = {"nnunet_task027_acdc": work / "nnunet_task027_acdc/ensemble",
               "nnunet_task114_mnms": work / "nnunet_task114_mnms/ensemble",
               "nnunet_ukb_sax": work / "nnunet_ukb_sax/2d"}
    for net in nets:
        for pid in pids:
            img, _ = seglib.ed_paths(pid)
            arr = np.asarray(nib.load(str(src_dir[net] / f"{pid}.nii.gz")).dataobj).astype(np.uint8)
            ref = nib.load(str(img))
            assert arr.shape == ref.shape, (net, pid, arr.shape, ref.shape)
            seglib.save_like(seglib.remap(arr, net), img, Path(a.out) / net / f"{pid}_ED.nii.gz")
    print(json.dumps({"timing_s": timing}))


if __name__ == "__main__":
    main()
