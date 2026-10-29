"""Known-answer and edge-case tests for code/manuscript/v03_analyses.py.
Run (project root): /project/home/p201509/envs/cvri-sim/bin/python -I tests/test_v03_analyses.py
(pytest is not installed in cvri-sim; the runner at the bottom calls every test_* function.)"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code', 'manuscript'))
import v03_analyses as V  # noqa: E402


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def mk_rows(spec):
    """spec: list of (patient, source, resolution, istar, ratio_v_t, ratio_v_endo)."""
    df = pd.DataFrame(spec, columns=['patient', 'source', 'resolution_mm', 'istar', 'ratio_v_t', 'ratio_v_endo'])
    df['group'] = 'NOR'
    df['construction'] = df.patient.astype(str) + df.source + df.resolution_mm.astype(str)
    return df


# ---------------------------------------------------------------- bootstrap helpers
def test_pct_ci_constant_and_empty():
    assert V.pct_ci([2.0] * 10) == [2.0, 2.0]
    lo, hi = V.pct_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_boot_median_known_and_deterministic():
    m, ci = V.boot_median([1, 2, 3, 4, 100])
    assert m == 3.0
    assert ci == V.boot_median([1, 2, 3, 4, 100])[1]  # same seed, same CI
    assert ci[0] <= m <= ci[1]
    m, ci = V.boot_median([5.0, np.nan, 5.0])
    assert m == 5.0 and ci == [5.0, 5.0]
    m, ci = V.boot_median([np.nan])
    assert np.isnan(m)


def test_boot_indices_seed():
    a = V.boot_indices(7)
    b = np.random.default_rng(20261028).integers(0, 7, (2000, 7))
    assert a.shape == (2000, 7) and (a == b).all()


def test_iqr():
    assert V.iqr([1, 2, 3, 4, 5]) == [2.0, 4.0]


# ---------------------------------------------------------------- per-heart summaries
def test_per_heart_network_known():
    rows = mk_rows([(1, 'a', r, i, x, 1.0) for (r, i), x in
                    zip([(1.5, 0), (1.5, 1), (1.5, 2), (1.0, 0), (1.0, 1), (1.0, 2)], [0.8, 0.9, 1.0, 1.1, 1.2, 1.5])])
    ph = V.per_heart_network(rows)
    assert close(ph.loc[1, 'ratio_v_t'], 1.05)
    # |r-1| = .2 .1 0 .1 .2 .5 -> median .15
    assert close(ph.loc[1, 'err_v_t'], 0.15)
    assert close(ph.loc[1, 'abserr_med_v_t'], 0.05)
    assert ph.loc[1, 'err_v_endo'] == 0.0 and ph.loc[1, 'n_est'] == 6


def test_heart_cv_known_ddof1_and_median_over_theta():
    # istar 0: values 1, 3 -> sd sqrt(2), mean 2 -> 0.7071; istar 1: 2, 2 -> 0; istar 2: 1, 2 -> 0.7071/1.5
    rows = mk_rows([(1, 'a', 1.5, 0, 1, 1), (1, 'b', 1.5, 0, 3, 1), (1, 'a', 1.5, 1, 2, 1), (1, 'b', 1.5, 1, 2, 1),
                    (1, 'a', 1.5, 2, 1, 1), (1, 'b', 1.5, 2, 2, 1)])
    cv = V.heart_cv(rows, 'ratio_v_t')
    assert close(cv[1], np.sqrt(0.5) / 1.5)
    sd = V.heart_sd(rows, 'ratio_v_t')
    assert close(sd[1], np.sqrt(0.5))


def test_heart_cv_single_construction_is_nan():
    rows = mk_rows([(1, 'a', 1.5, 0, 1.0, 1.0)])
    assert np.isnan(V.heart_cv(rows, 'ratio_v_t')[1])
    assert np.isnan(V.heart_sd(rows, 'ratio_v_t')[1])


# ---------------------------------------------------------------- Kendall W
def test_kendall_w_perfect_and_null():
    R = np.array([[1, 2, 3, 4]] * 5)
    assert close(V.kendall_w(R), 1.0)
    assert close(V.kendall_w(np.array([[1, 2, 3], [3, 2, 1]])), 0.0)


def test_kendall_w_textbook():
    # m = 3 raters, n = 3 objects: ranks [1,2,3],[1,3,2],[2,1,3]; R = [4,6,8], S = 8, W = 12*8/(9*24) = 0.4444
    R = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3]])
    assert close(V.kendall_w(R), 96 / 216)


def test_kendall_w_ties_correction():
    # all tied: denominator m^2(n^3-n) - m*sum(t^3-t) = 0 -> NaN
    assert np.isnan(V.kendall_w(np.array([[2.5] * 4, [2.5] * 4])))
    # one tie: ranks [1.5,1.5,3],[1,2,3]; R=[2.5,3.5,6], S=6.5; T=6; den=4*24-2*6=84; W=78/84
    assert close(V.kendall_w(np.array([[1.5, 1.5, 3], [1, 2, 3]])), 78 / 84)
    assert np.isnan(V.kendall_w(np.array([[1, 2, 3]])))


def test_rank_largest_first_and_first_counts():
    m = np.array([[0.1, 0.5, 0.3], [0.2, 0.2, 0.1]])
    r = V.rank_largest_first(m)
    assert (r[0] == [3, 1, 2]).all() and (r[1] == [1.5, 1.5, 3]).all()
    assert V.first_counts(m, ['a', 'b', 'c']) == {'a': 1, 'b': 2, 'c': 0}


def test_friedman_from_w():
    chi2, p = V.friedman_from_w(1.0, 10, 4)
    assert close(chi2, 30.0) and p < 1e-5
    chi2, p = V.friedman_from_w(0.0, 10, 4)
    assert chi2 == 0 and close(p, 1.0)


def test_w_permutation_and_boot():
    R = np.array([[1, 2, 3, 4]] * 20, float)
    p, wnull = V.w_permutation_p(R, nperm=200)
    assert p < 0.01 and wnull < 0.2
    assert V.boot_w(R, nb=50) == [1.0, 1.0]


def test_friedman_matches_scipy_random():
    from scipy import stats
    x = np.random.default_rng(1).random((30, 4))
    r = V.rank_largest_first(x)
    chi2, p = V.friedman_from_w(V.kendall_w(r), *r.shape)
    ref = stats.friedmanchisquare(*x.T)
    assert close(chi2, ref.statistic, 1e-9) and close(p, ref.pvalue, 1e-9)


# ---------------------------------------------------------------- Spearman
def test_spearman_known_and_edges():
    assert close(V.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert close(V.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert np.isnan(V.spearman([1, 1, 1], [1, 2, 3]))
    assert np.isnan(V.spearman([1, 2], [1, 2]))
    assert close(V.spearman([1, 2, 3, np.nan], [1, 2, 3, 9]), 1.0)


def test_pooled_and_within_heart_spearman():
    df = pd.DataFrame(dict(patient=[1] * 4 + [2] * 4, x=[1, 2, 3, 4, 5, 6, 7, 8], y=[8, 7, 6, 5, 4, 3, 2, 1]))
    est, ci, nh, npair = V.pooled_spearman_boot(df, 'x', 'y', nb=100)
    assert close(est, -1.0) and nh == 2 and npair == 8
    # resamples drawing one heart twice still give -1; all finite resamples equal -1
    assert close(ci[0], -1.0) and close(ci[1], -1.0)
    wh = V.within_heart_spearman(df, 'x', 'y')
    assert close(wh[1], -1.0) and close(wh[2], -1.0)
    # pooled can differ from within-heart: between-heart offset reverses the sign
    df2 = pd.DataFrame(dict(patient=[1] * 3 + [2] * 3, x=[1, 2, 3, 11, 12, 13], y=[3, 2, 1, 13, 12, 11]))
    assert V.spearman(df2.x, df2.y) > 0
    assert (V.within_heart_spearman(df2, 'x', 'y') == -1).all()


def test_boot_mean():
    m, ci, n = V.boot_mean([1.0, np.nan, 3.0])
    assert m == 2.0 and n == 2 and 1.0 <= ci[0] <= ci[1] <= 3.0


# ---------------------------------------------------------------- ensemble size
def test_subset_cv_matrix_and_ensemble_stat():
    nets = ('a', 'b', 'c')
    vals = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    spec = [(1, s, r, i, vals[s], 1.0) for s in nets for r in (1.5, 1.0) for i in (0, 1)]
    rows = mk_rows(spec)
    m2, subs = V.subset_cv_matrix(rows, nets, 2, 'ratio_v_t')
    assert len(subs) == 3 and list(m2.columns) == ['a+b', 'a+c', 'b+c']
    # a+b: values 1,1,2,2 -> sd = sqrt(1/3), mean 1.5
    assert close(m2.loc[1, 'a+b'], np.sqrt(1 / 3) / 1.5)
    m3, _ = V.subset_cv_matrix(rows, nets, 3, 'ratio_v_t')
    assert close(m3.loc[1, 'a+b+c'], np.std([1, 1, 2, 2, 3, 3], ddof=1) / 2)
    assert close(V.ensemble_stat(np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 4.0]])), (2.0 + 4.0) / 2)


def test_boot_ensemble_ratio_and_half_spread():
    mats = {2: np.array([[1.0, 1.0]] * 5), 4: np.array([[2.0]] * 5)}
    bo = V.boot_ensemble(mats, nb=50)
    assert bo[2]['ratio_to_k4'] == 0.5 and bo[2]['ratio_ci'] == [0.5, 0.5] and bo[4]['ratio_to_k4'] == 1.0
    per, allf, anyf = V.half_spread_fraction(np.array([[1.0, 0.9], [1.0, 1.0]]), np.array([2.0, 2.0]))
    assert (per == [0.5, 1.0]).all() and allf == 0.5 and anyf == 1.0
    # boundary: exactly half counts
    per, _, _ = V.half_spread_fraction(np.array([[1.0]]), np.array([2.0]))
    assert per[0] == 1.0


# ---------------------------------------------------------------- regression on archived data (small fixture)
def test_regression_E1_and_heart101():
    e = V.load_estimates(os.path.join(ROOT, V.INPUTS['estimates']))
    prim = e[e.include_primary]
    cv = V.heart_cv(prim, 'ratio_v_t')
    assert round(100 * float(np.median(cv)), 1) == 16.1
    assert round(100 * float(np.median(V.heart_cv(prim, 'ratio_v_endo'))), 1) == 2.6
    h = prim[prim.patient == 101]
    assert h.construction.nunique() == 10 and len(h) == 30
    ph = V.per_heart_network(V.network_rows(h, 'nnunet_task027_acdc'))
    r = h[h.source == 'nnunet_task027_acdc'].ratio_v_t
    assert close(ph.loc[101, 'ratio_v_t'], float(np.median(r)))
    assert len(V.network_rows(h, 'expert_1.0')) == 3


if __name__ == '__main__':
    os.chdir(ROOT)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    fails = 0
    for n, f in tests:
        try:
            f()
            print('PASS', n)
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print('FAIL', n, repr(ex))
    print(f'{len(tests) - fails} passed, {fails} failed')
    sys.exit(1 if fails else 0)
