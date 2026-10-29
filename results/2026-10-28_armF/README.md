# Arm F, model-form comparator (I11 full run), 2026-10-28

Project root `/project/home/p201509/work/cvri/simulation`. Owner armF. Governs: `final/AMENDMENT_2026-10-28.md` (A4),
`final/STUDY_PLAN.md` section 3 (arm F). Code: `code/full/armF/` (definitions in `armf_lib.py` docstring).

## Design
- Hearts (A4 rule, `MANIFEST.tsv` of `data/processed/acdc_mesh_v2`, expert 1.5 mm rv 3.0 reference passing G2):
  DCM 101, 106; HCM 104, 105; MINF 103, 112; NOR 102, 107; RV 109, 119.
- Construction: expert, 1.5 mm, A2 layer (`etag`), RV wall 3 mm. θ*: 3 per heart, seed 20261028 + id (reproduces the
  gates draws for 101, 102, 104 exactly; `test_armf.py`).
- Targets `re`: openCARP v20.0 RE+ (dream.solve 1), monodomain, tenTusscherPanfilov, default conductivities, dt 25 µs,
  Vm every 0.5 ms, layout default0, renumbering 0, run to the unit tmax; ECG = −W·Vm minus its t = 0 value, same W.
- Control `tmpl`: pure-eikonal target with the activation-template ECG (the arm C target), same emulator.
- Emulator: 9 × 9 log grid + kriged refinement (88 refinement runs in total, round 2 added none). Losses: corr (primary),
  amp. Windows: `qrs` (t ≤ max target activation + 10 ms; primary for arm F, assumption, see below), `full` (0 to tmax).
- Summary: per heart the mean over 3 θ* of log(θ̂/θ*); population geometric mean (GM) with t 95% CI over 10 hearts
  (df 9) and cluster-bootstrap percentile CI (2,000 resamples, seed 20261028).

## Results (1.5 mm, round 2; `summary_armF.json`, `estimates_armF.csv`)
| set, loss, window | v_endo GM [t 95% CI] {bootstrap} | v_t GM [t 95% CI] {bootstrap} |
|---|---|---|
| re, corr, qrs (primary) | 0.971 [0.967, 0.974] {0.968, 0.973} | 0.929 [0.909, 0.950] {0.912, 0.945} |
| re, corr, full | 0.971 [0.968, 0.974] | 0.921 [0.897, 0.945] |
| re, amp, qrs | 0.961 [0.954, 0.969] | 0.943 [0.914, 0.974] |
| re, amp, full | 0.961 [0.953, 0.969] | 0.944 [0.914, 0.975] |
| tmpl (control), all four | 1.000 (all 30 within 1e-4) | 1.000 |

- Primary, all 30 estimates: v_endo median 0.971 (2.5th to 97.5th percentile 0.952 to 0.984), SD of log 0.009;
  v_t median 0.930 (0.876 to 0.977), SD of log 0.034. Per-heart v_t GM 0.878 (109) to 0.968 (105).
- Paired RE/control contrast equals the RE ratios to three decimals (v_endo 0.971 [0.967, 0.974], v_t 0.929 [0.909, 0.950]).
- Bound hits: 0 of 120 RE estimates at 1.5 mm (and 0 for control, dt and 1.0 mm sets). Censoring sensitivity identical.
- Fit loss (primary): median 0.0053 (0.0021 to 0.0135); loss at θ̂ minus loss at θ* median −0.0093 (θ* is not the optimum).
- RE vs template ECG at θ* (QRS window, 240 lead values): r median 0.986 (0.951 to 0.996), minimum 0.932; per-target mean
  r median 0.985 (minimum 0.969); amplitude ratio RE/template median 1.059 (1.034 to 1.070); relative L2 median 0.141.
- RE act.igb equals the eikonal activation exactly (difference 0 ms at every node); the Vm upstroke (−40 mV) lags it by
  a median 1.21 ms (1.19 to 1.22). The model-form difference is the waveform, not the activation sequence.
- RE wall time per run (1 core, packed 36 on one node): median 205 s (108 to 275) at 1.5 mm; 751 to 792 s at 1.0 mm.

## Convergence (heart 101)
- dt 12.5 µs vs 25 µs (3 θ*): ECG min lead r ≥ 0.99998, relative L2 0.0017 to 0.0021, activation identical;
  θ̂ ratio (12.5/25) 0.99989 to 0.99999 (v_endo), 0.99942 to 0.99988 (v_t). Time step converged for this purpose.
- Mesh 1.0 mm (210,961 nodes; own 1.0 mm emulator, W and control): bias v_endo GM 0.977 and v_t 0.959 against 0.974 and
  0.942 at 1.5 mm (n = 3 θ*, one heart; no CI). Control at 1.0 mm 1.000. The bias in v_t is mesh-dependent at about
  1.7 percentage points on this heart. RE ECG 1.0 vs 1.5 mm min lead r 0.54 to 0.82: that compares two different
  constructions (mesh factor), not RE convergence.

## Jobs, versions, seeds
- 5394178 test (qos test), 00:16:21, 0.27 node-hours; 5394247 full (qos default), 00:31:07, 0.52 node-hours. Total 0.79.
- openCARP v20.0 AppImage (`carpio.OPENCARP`); Python 3.11.17, numpy 2.4.6, scipy 1.17.1, h5py 3.16.0 (env cvri-sim).
- Seeds: θ* 20261028 + id; bootstrap 20261028; fibres pre-built seeded upstream. No other stochastic step.
- Code SHA-256: see `MANIFEST`; job logs in `logs/` (hashes printed at job start).

## Assumptions and deviations
1. Primary window `qrs` is an arm F choice: the RE target contains early repolarisation that the template cannot
   represent. `full` is reported; the v_t bias differs by 0.008.
2. The emulator is refined at θ̂ of all target sets (up to 12 points), not the gates cap of 9.
3. test_armf sigmoid-template agreement threshold (r > 0.995) was set after seeing r = 0.9984; the exact checks are the
   constant and step cases.
4. Mesh check is one heart and 3 θ*, without CI.
