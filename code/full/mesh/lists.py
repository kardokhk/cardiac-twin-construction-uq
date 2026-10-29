"""Construction lists for the full arm (owner mesh2, 2026-10-28). Usage: python -I -B lists.py {rvhearts|todo} [test]
rvhearts: the 10 RV-wall sensitivity hearts = 2 per ACDC group, lowest patient ids whose expert 1.5 mm construction
passed G2 (results/2026-10-28_G2_meshing/constructions.tsv, status pass); the same rule as arm F (Amendment A4).
todo: one line 'pid source vox rv_wall' per non-reference construction (1.0 mm first)."""
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import acdc_lib as A

PRIMARY = ['expert', 'nnunet_task027_acdc', 'nnunet_task114_mnms', 'nnunet_ukb_sax', 'cinema_acdc_sax']
SENS = ['ukbb_cardiac_fcn_sa']
TRIO = [101, 102, 104]


def rv_hearts():
    d = pd.read_csv(os.path.join(A.ROOT, 'results', '2026-10-28_G2_meshing', 'constructions.tsv'), sep='\t')
    g = d[(d.source == 'expert') & (d.vox == 1.5) & (d.status == 'pass')].sort_values('patient')
    return sorted(int(p) for grp, x in g.groupby('group') for p in x.patient.head(2))


def todo(test=False):
    hearts = TRIO if test else list(range(101, 151))
    out = []
    for vox in (1.0, 1.5):
        for p in hearts:
            for s in PRIMARY + SENS:
                if s == 'expert' and vox == 1.5:
                    continue          # the reference construction is built by 'ref'
                out.append((p, s, vox, 3.0))
    rvh = [101] if test else rv_hearts()
    for p in rvh:
        for s in (PRIMARY[:2] if test else PRIMARY):
            for w in (2.0, 4.0):
                out.append((p, s, 1.5, w))
    if not test:
        # extra, not pre-registered: the same RV-wall levels at 1.0 mm, added after the test job showed that a 2 mm
        # wall disconnects (F3) at 1.5 mm voxels
        for p in rvh:
            for s in PRIMARY:
                for w in (2.0, 4.0):
                    out.insert(0, (p, s, 1.0, w))
    return out


if __name__ == '__main__':
    test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    if sys.argv[1] == 'rvhearts':
        print(' '.join(map(str, rv_hearts())))
    else:
        for p, s, v, w in todo(test):
            print(p, s, v, w)
