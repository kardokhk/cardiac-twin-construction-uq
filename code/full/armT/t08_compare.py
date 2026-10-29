"""Arm T follow-up 2026-10-28 (owner armT): old (analysis.json, pooled refinement, round 1) versus new (analysis_v2.json,
per-arm refinement, round 2) for every cell, every S5 item, the own-torso check, and the checks of t07 (checks_v2.json).
Writes results/2026-10-28_armT/compare_v2.md. Usage: python -I t08_compare.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armt_lib as L          # noqa: E402

A = json.load(open(os.path.join(L.OUT, 'analysis.json')))
B = json.load(open(os.path.join(L.OUT, 'analysis_v2.json')))
CK = json.load(open(os.path.join(L.OUT, 'checks_v2.json')))


def s(v):
    return f'{v[0]*100:+.1f}% [{v[1]*100:+.1f}, {v[2]*100:+.1f}]'


def a_(v):
    return f'{v[0]*100:.1f}% [{v[1]*100:.1f}, {v[2]*100:.1f}]'


key = lambda c: (c['loss'], c['ref'], c['reg'], c['set'])  # noqa: E731
Bc = {key(c): c for c in B['cells']}
out = ['# Arm T old versus new (pooled refinement round 1 -> per-arm refinement round 2), 2026-10-28', '',
       'Made by `code/full/armT/t08_compare.py` from `analysis.json` (old) and `analysis_v2.json` (new).', '',
       '## Own-torso known answer (max |theta_hat/theta* - 1| over 66 estimates per loss; median in brackets)', '',
       '| version | round | loss | v_endo max [median] | v_t max [median] | bound hits |', '|---|---|---|---|---|---|']
for o in A['own']:
    out.append(f"| old (pooled refinement) | 1 | {o['loss']} | {o['max_abs_rel_v_endo']:.2e} | {o['max_abs_rel_v_t']:.2e} | {o['bound_hits']} |")
for o in B['own_by_round']:
    lab = 'new, no refinement' if o['round'] == 0 else f"new, own refinement only"
    out.append(f"| {lab} | {o['round']} | {o['loss']} | {o['max_abs_rel_v_endo']:.2e} [{o['median_abs_rel_v_endo']:.2e}] | "
               f"{o['max_abs_rel_v_t']:.2e} [{o['median_abs_rel_v_t']:.2e}] | {o['bound_hits']} |")
out += ['', '## Bias cells (old -> new)', '',
        '| loss | ref | reg | set | v_endo signed | v_endo abs | v_t signed | v_t abs | bound hits | fit loss median [IQR] |',
        '|---|---|---|---|---|---|---|---|---|---|']
for c in A['cells']:
    d = Bc[key(c)]
    fl = lambda x: f"{x['fval_median']:.3f} [{x['fval_q25']:.3f}, {x['fval_q75']:.3f}]"  # noqa: E731
    out.append(f"| {c['loss']} | {c['ref']} | {c['reg']} | {c['set']} | {s(c['v_endo_signed'])} -> {s(d['v_endo_signed'])} | "
               f"{a_(c['v_endo_abs'])} -> {a_(d['v_endo_abs'])} | {s(c['v_t_signed'])} -> {s(d['v_t_signed'])} | "
               f"{a_(c['v_t_abs'])} -> {a_(d['v_t_abs'])} | {c['bound_hits']}/{c['n_estimates']} -> "
               f"{d['bound_hits']}/{d['n_estimates']} | {fl(c)} -> {fl(d)} |")
out += ['', '## S5 (old -> new)', '', '| item | ref | reg | loss | parameter | old | new |', '|---|---|---|---|---|---|---|']
for c, d in zip(A['s5'], B['s5']):
    assert (c['what'], c['ref'], c.get('reg'), c.get('loss'), c.get('param')) == \
           (d['what'], d['ref'], d.get('reg'), d.get('loss'), d.get('param'))
    v, w = c.get('rho', c.get('diff')), d.get('rho', d.get('diff'))
    sc = 100 if 'diff' in c else 1
    fm = (lambda x: f'{x[0]*sc:+.1f} [{x[1]*sc:+.1f}, {x[2]*sc:+.1f}]') if sc == 100 else \
         (lambda x: f'{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]')
    out.append(f"| {c['what']} | {c['ref']} | {c.get('reg', 'icp - trans')} | {c.get('loss', '-')} | {c.get('param', '-')} | "
               f"{fm(v)} | {fm(w)} |")
out += ['', '## Identity checks through the ICP and translation-only paths', '',
        '| heart | case | finite | ICP rotation deg | ICP t mm | ICP rel W | trans shift mm | trans rel W |', '|---|---|---|---|---|---|---|---|']
for h in L.REFS:
    r = CK.get(f'identity_{h}', {})
    for k, v in r.items():
        if isinstance(v, dict) and 'finite' in v:
            if v['finite']:
                out.append(f"| {h} | {k} | yes | {v['rot_deg']:.2e} | {v['t_norm_mm']:.2e} | {v['icp_rel_W']:.1e} | "
                           f"{v['trans_shift_mm']:.2e} | {v['trans_rel_W']:.1e} |")
            else:
                out.append(f'| {h} | {k} | no (NaN) | - | - | - | - | - |')
out += ['', '## eplib.icp versus the d <= p90 trim on the 84 real registrations', '']
mx = {}
for k, v in CK.items():
    if k.startswith('icpeq_'):
        for rid, r in v.items():
            if isinstance(r, dict):
                for q, x in r.items():
                    if not q.startswith('rms'):
                        mx[q] = max(mx.get(q, 0.0), x)
out += [f'- max {q}: {x:.2e}' for q, x in sorted(mx.items())]
out += ['', '## KCL axis convention, translation-only lead r (median over theta*)', '',
        '| heart | ref | LPS as used | RAS | LAS mirror | RPS mirror | anterior/left checks under RAS (mm) | LPS reproduces stored W |',
        '|---|---|---|---|---|---|---|---|']
for k in sorted(x for x in CK if x.startswith('axes_')):
    v = CK[k]
    for rid in L.REFS:
        r = v[rid]
        out.append(f"| {v['heart']} | {rid} | {r['LPS_as_used']['lead_r_median']:+.3f} | {r['RAS']['lead_r_median']:+.3f} | "
                   f"{r['LAS_mirror']['lead_r_median']:+.3f} | {r['RPS_mirror']['lead_r_median']:+.3f} | "
                   f"{r['RAS']['heart_anterior_of_body_mm']:+.1f} / {r['RAS']['lvblood_left_of_rvblood_mm']:+.1f} | "
                   f"{r['LPS_as_used']['reproduces_stored_W']:.1e} |")
open(os.path.join(L.OUT, 'compare_v2.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
