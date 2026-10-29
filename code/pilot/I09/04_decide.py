"""I09 pilot step 4: gather per-case JSON rows into one HDF5 table and apply the pre-registered criteria.

Pre-registered (critique/I09-redteam.md s6 and the pilot specification, 2026-10-22; lines not moved):
  KILL-1  R^2(TAT ~ LVEDV + LV mass + RVEDV), OLS with intercept, Gaussian arm, >= 0.90.
  KILL-2  the linear rule fitted on the Gaussian arm predicts the uniform-minus-Gaussian TAT shift inside the
          bootstrap 95% CI of the simulated shift. Red-team wording: mean AND q95; specification wording: "the TAT shift"
          (mean). Both readings are reported.
Predicted mean shift = beta_hat . (xbar_U - xbar_G). Predicted q95 shift = q95{beta.x_U,i + e_j} -
q95{beta.x_G,i + e_j}, e_j the Gaussian-arm OLS residuals (all i, j pairs). Bootstrap: 2,000 resamples, each arm
resampled independently, percentile intervals, seed 20261022. Primary pipeline decides; secondary reported.
Usage: python -I 04_decide.py <rows_dir> <weights.csv> <out_dir>
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

rows_dir, wfile, out = sys.argv[1:4]
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(rows_dir, "*.json")))]
df = pd.DataFrame(rows)
df["arm"] = np.where(df["case"].str.startswith("G"), "gauss", "uniform")
W = np.loadtxt(wfile, delimiter=",")
df = df.sort_values("row").reset_index(drop=True)

# one HDF5 table
num = [c for c in df.columns if c not in ("case", "arm", "laplace_iters") and df[c].dtype != object]
with h5py.File(os.path.join(out, "I09_pilot_table.h5"), "w") as h:
    g = h.create_group("cases")
    g.create_dataset("case", data=df["case"].astype("S8").values)
    g.create_dataset("arm", data=df["arm"].astype("S8").values)
    for c in num:
        g.create_dataset(c, data=df[c].astype(float).values)
    g.create_dataset("weights", data=W[df["row"].values])
    h.attrs["note"] = ("I09 pilot, KIT SSM (Zenodo 4506463) instances; rows 0-199 Gaussian seed 20261022, "
                       "200-299 deposited uniform weights. Script code/pilot/I09/04_decide.py")
df.drop(columns=["laplace_iters"], errors="ignore").to_csv(os.path.join(out, "I09_pilot_table.csv"), index=False)

rng = np.random.default_rng(20261022)
B = 2000
XCOLS = ["LVEDV_mL", "LVmass_g", "RVEDV_mL"]
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)


failed = df[~df["ok"].astype(bool)]
P(f"cases {len(df)}; failed {len(failed)}")
for _, r in failed.iterrows():
    w = W[int(r["row"])]
    P(f"  FAILED {r['case']} row {int(r['row'])} |w|^2 {w @ w:.1f} max|w| {np.abs(w).max():.2f} "
      f"inverted {r.get('n_inverted')} rc {r.get('rc_primary')}")
P("inverted-tet meshes:", df.loc[df["n_inverted"] > 0, ["case", "n_inverted"]].values.tolist())


def ols(X, y):
    Xd = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    e = y - Xd @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, e, r2


def q95conv(Xu, b, e):
    f = np.column_stack([np.ones(len(Xu)), Xu]) @ b
    return np.quantile((f[:, None] + e[None, :]).ravel(), 0.95)


def ci(a):
    return np.quantile(a, [0.025, 0.975])


summary = {}
for pipe in ("primary", "secondary"):
    y = f"TAT_{pipe}_ms"
    if y not in df:
        continue
    ok = df["ok"].astype(bool) & df[y].notna()
    G = df[ok & (df.arm == "gauss")]; U = df[ok & (df.arm == "uniform")]
    XG, yG, XU, yU = G[XCOLS].values, G[y].values, U[XCOLS].values, U[y].values
    b, e, r2 = ols(XG, yG)
    sd_res = np.sqrt(e @ e / (len(yG) - 4))
    d_mean_obs = yU.mean() - yG.mean()
    d_q95_obs = np.quantile(yU, 0.95) - np.quantile(yG, 0.95)
    d_mean_pred = b[1:] @ (XU.mean(0) - XG.mean(0))
    d_q95_pred = q95conv(XU, b, e) - q95conv(XG, b, e)
    bs = {k: [] for k in ("r2", "dmo", "dqo", "dmp", "dqp")}
    for _ in range(B):
        ig = rng.integers(0, len(yG), len(yG)); iu = rng.integers(0, len(yU), len(yU))
        bb, ee, rr = ols(XG[ig], yG[ig])
        bs["r2"].append(rr)
        bs["dmo"].append(yU[iu].mean() - yG[ig].mean())
        bs["dqo"].append(np.quantile(yU[iu], 0.95) - np.quantile(yG[ig], 0.95))
        bs["dmp"].append(bb[1:] @ (XU[iu].mean(0) - XG[ig].mean(0)))
        bs["dqp"].append(q95conv(XU[iu], bb, ee) - q95conv(XG[ig], bb, ee))
    c = {k: ci(np.array(v)) for k, v in bs.items()}
    # out-of-sample residuals on the uniform arm
    eU = yU - np.column_stack([np.ones(len(yU)), XU]) @ b
    # supplementary fits
    _, _, r2_lvedv = ols(XG[:, :1], yG)
    wG = W[G["row"].values]
    _, _, r2_modes = ols(np.column_stack([XG, wG[:, :10]]), yG)
    _, _, r2_U = ols(XU, yU)
    k1 = r2 >= 0.90
    in_m = c["dmo"][0] <= d_mean_pred <= c["dmo"][1]
    in_q = c["dqo"][0] <= d_q95_pred <= c["dqo"][1]
    k2_rt = in_m and in_q
    P(f"\n=== {pipe} pipeline: n_G {len(yG)}, n_U {len(yU)}")
    P(f"TAT Gaussian mean (SD) {yG.mean():.2f} ({yG.std(ddof=1):.2f}) ms; uniform {yU.mean():.2f} "
      f"({yU.std(ddof=1):.2f}) ms")
    P(f"phenotypes Gaussian mean (SD): " + ", ".join(f"{k} {G[k].mean():.1f} ({G[k].std(ddof=1):.1f})" for k in XCOLS))
    P(f"phenotypes uniform  mean (SD): " + ", ".join(f"{k} {U[k].mean():.1f} ({U[k].std(ddof=1):.1f})" for k in XCOLS))
    P(f"OLS beta (intercept, LVEDV, mass, RVEDV): {np.round(b, 4).tolist()}")
    P(f"R2 Gaussian = {r2:.3f} (95% CI {c['r2'][0]:.3f} to {c['r2'][1]:.3f}); KILL-1 (>=0.90): {k1}")
    P(f"residual SD after linear rule (Gaussian, in-sample) = {sd_res:.2f} ms; uniform out-of-sample RMS "
      f"{np.sqrt((eU**2).mean()):.2f} ms, mean residual {eU.mean():.2f} ms")
    P(f"mean shift U-G observed {d_mean_obs:.2f} ms (95% CI {c['dmo'][0]:.2f} to {c['dmo'][1]:.2f}); "
      f"predicted {d_mean_pred:.2f} (95% CI {c['dmp'][0]:.2f} to {c['dmp'][1]:.2f}); inside obs CI: {in_m}")
    P(f"q95 shift U-G observed {d_q95_obs:.2f} ms (95% CI {c['dqo'][0]:.2f} to {c['dqo'][1]:.2f}); "
      f"predicted {d_q95_pred:.2f} (95% CI {c['dqp'][0]:.2f} to {c['dqp'][1]:.2f}); inside obs CI: {in_q}")
    P(f"KILL-2 red-team reading (mean AND q95 inside): {k2_rt}; specification reading (mean inside): {in_m}")
    P(f"supplementary: R2 on LVEDV only {r2_lvedv:.3f}; R2 on 3 phenotypes + modes 1-10 {r2_modes:.3f}; "
      f"R2 within uniform arm {r2_U:.3f}")
    summary[pipe] = dict(n_G=len(yG), n_U=len(yU), R2=r2, R2_ci=c["r2"].tolist(), sd_res=sd_res,
                         d_mean_obs=d_mean_obs, d_mean_obs_ci=c["dmo"].tolist(), d_mean_pred=d_mean_pred,
                         d_mean_pred_ci=c["dmp"].tolist(), d_q95_obs=d_q95_obs, d_q95_obs_ci=c["dqo"].tolist(),
                         d_q95_pred=d_q95_pred, d_q95_pred_ci=c["dqp"].tolist(), kill1=bool(k1),
                         kill2_redteam=bool(k2_rt), kill2_mean_only=bool(in_m), r2_lvedv=r2_lvedv,
                         r2_modes=r2_modes, r2_uniform=r2_U, beta=b.tolist())
P("\nother: median t_total_s", df["t_total_s"].median().round(1), "max", df["t_total_s"].max().round(1),
  "| helix endo/mid/epi median", df["helix_endo_deg"].median().round(1), df["helix_mid_deg"].median().round(1),
  df["helix_epi_deg"].median().round(1))
json.dump(summary, open(os.path.join(out, "decision.json"), "w"), indent=1, default=float)
open(os.path.join(out, "decision.txt"), "w").write("\n".join(lines) + "\n")
