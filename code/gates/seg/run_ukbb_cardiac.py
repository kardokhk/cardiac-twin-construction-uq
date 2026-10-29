"""ukbb_cardiac FCN_sa (Bai et al. 2018) on ACDC ED frames, TF1 graph run through tf.compat.v1 in TF2 (CPU).

Follows ukbb_cardiac common/deploy_network.py (commit f465832c) exactly for one 3D frame: intensity rescale to the
1st and 99th percentiles, zero-pad x and y to multiples of 16 (centred), feed slices as N x X x Y x 1, read 'pred:0'.
No resampling (the original applies the network at native in-plane resolution). Labels 1 LV, 2 Myo, 3 RV are mapped
to the ACDC convention by seglib.remap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seglib  # noqa: E402

MODEL = "/project/home/p201509/work/cvri/simulation/data/raw/weights/ukbb_cardiac_fcn_sa/FCN_sa"


def rescale_intensity(image, thres=(1.0, 99.0)):
    val_l, val_h = np.percentile(image, thres)
    image2 = np.clip(image, val_l, val_h)
    return (image2.astype(np.float32) - val_l) / (val_h - val_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tf.compat.v1.disable_eager_execution()
    t0 = time.time()
    log = []
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(MODEL + ".meta")
        saver.restore(sess, MODEL)
        for pid in seglib.pid_list(a.pids):
            img_path, _ = seglib.ed_paths(pid)
            image = np.asarray(nib.load(str(img_path)).dataobj).astype(np.float32)
            X, Y = image.shape[:2]
            image = rescale_intensity(image, (1, 99))
            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), "constant")
            inp = np.expand_dims(np.transpose(image, axes=(2, 0, 1)).astype(np.float32), -1)
            pred = sess.run("pred:0", feed_dict={"image:0": inp, "training:0": False})
            pred = np.transpose(pred, axes=(1, 2, 0))[x_pre:x_pre + X, y_pre:y_pre + Y]
            lab = seglib.remap(pred.astype(np.uint8), "ukbb_cardiac_fcn_sa")
            seglib.save_like(lab, img_path, Path(a.out) / "ukbb_cardiac_fcn_sa" / f"{pid}_ED.nii.gz")
            log.append(pid)
    print(json.dumps({"tf": tf.__version__, "seconds": time.time() - t0, "cases": log}))


if __name__ == "__main__":
    main()
