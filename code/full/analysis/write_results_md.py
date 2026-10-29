"""Render results.json (from run_analysis.py) as RESULTS.md: one table per estimand. No prose beyond definitions."""


def pc(x, d=1):
    return 'nan' if x is None or x != x else f'{100 * x:.{d}f}%'


def pci(ci, d=1):
    return f'[{pc(ci[0], d)}, {pc(ci[1], d)}]'


def f3(x):
    return 'nan' if x is None or x != x else f'{x:.3f}'


def lev(x, unit=None):
    if isinstance(x, str):
        return x
    if x is None or x != x:
        return 'nan'
    return f'{x:.1f}'


TLAB = {'heart': 'heart-level mean construction offset plus between-heart variation (random heart intercept)'}


E2D_TERMS = [('segmentation_S', 'segmentation S (fixed)'), ('mesh_M', 'mesh M (fixed)'), ('SxM', 'S x M (fixed)'),
             ('theta_star', 'theta* (fixed)'), ('heart', 'heart (random intercept)'),
             ('heart_source', 'heart:source (random)'), ('residual', 'residual')]
E2D_DERIVED = [('within_heart_construction', 'within-heart construction share, (S + M + SxM + heart:source) / (same + residual)'),
               ('within_heart_construction_fixed_only', 'fixed S, M, SxM only, / (S + M + SxM + heart:source + residual)'),
               ('source_of_construction', 'source part of construction, (S + heart:source) / construction'),
               ('heart_source_of_construction', 'heart:source part of construction, heart:source / construction')]


def e2d_lines(D):
    cells = [(k, p) for k in D['sets'] for p in ['v_endo', 'v_t']]
    head = ' | '.join(f'{"corr" if k == "primary_no_ref" else "amp"} {p}' for k, p in cells)
    L = ['### E2d. ' + D['label'], '',
         f'Model: `{D["model"]}`. Rows: {D["rows"]}. {D["bootstrap"]}. Share = {D["definitions"]["share"]}. '
         f'Term variance: {D["definitions"]["term_variance"]}. Boundary fit: {D["definitions"]["boundary"]}. '
         f'Aliasing: {D["definitions"]["aliasing"]}. corr = correlation loss (primary), amp = amplitude loss (secondary).', '']
    b0 = D['sets']['primary_no_ref']['v_t']
    L += [f'n = {b0["n_hearts"]} hearts, {b0["n_constructions"]} constructions, {b0["n_heart_source_groups"]} heart:source groups, '
          f'{b0["n_rows"]} estimates per loss; dropped aliased column: {", ".join(b0["dropped_aliased_columns"])}.', '',
          '| term | ' + head + ' |', '|---|' + '---|' * len(cells)]
    for t, lab in E2D_TERMS + E2D_DERIVED:
        L.append(f'| {lab} | ' + ' | '.join(f'{pc(D["sets"][k][p]["shares"][t]["share"])} {pci(D["sets"][k][p]["shares"][t]["ci"])}'
                                            for k, p in cells) + ' |')
    L += ['', 'Variance components on the log scale [95% CI]:', '', '| component | ' + head + ' |', '|---|' + '---|' * len(cells)]
    for t, lab in E2D_TERMS:
        L.append(f'| {lab} | ' + ' | '.join(f'{D["sets"][k][p]["variance"][t]["value"]:.5f} [{D["sets"][k][p]["variance"][t]["ci"][0]:.5f}, '
                                            f'{D["sets"][k][p]["variance"][t]["ci"][1]:.5f}]' for k, p in cells) + ' |')
    fl = ['error', 'optimizer_not_converged', 'any_start_not_converged', 'rank_deficient', 'boundary_heart',
          'boundary_heart_source', 'boundary_either']
    L += ['', 'Fit flags. Point fit: ' + '; '.join(
        f'{"corr" if k == "primary_no_ref" else "amp"} {p} converged {D["sets"][k][p]["point_flags"]["optimizer_converged"]}, boundary '
        f'heart {D["sets"][k][p]["point_flags"]["boundary_heart"]}, heart:source {D["sets"][k][p]["point_flags"]["boundary_heart_source"]}'
        for k, p in cells) + '. Bootstrap fits (counts of ' + ', '.join(str(D['sets'][k][p]['n_boot_ok']) for k, p in cells) + ' fits):', '',
        '| flag | ' + head + ' |', '|---|' + '---|' * len(cells)]
    for f in fl:
        L.append(f'| {f} | ' + ' | '.join(str(D['sets'][k][p]['boot_flags'][f]) for k, p in cells) + ' |')
    xc = D.get('statsmodels_check', {})
    if 'error' in xc:
        L += ['', f'statsmodels cross-check failed: {xc["error"]}.']
    else:
        L += ['', 'Cross-check against statsmodels MixedLM (re_formula 1, vc_formula heart:source, same fixed design, REML): ' + '; '.join(
            f'{c}: s2 heart {v["own"]["s2_heart"]:.5f} vs {v["statsmodels"]["s2_heart"]:.5f}, heart:source '
            f'{v["own"]["s2_heart_source"]:.5f} vs {v["statsmodels"]["s2_heart_source"]:.5f}, residual {v["own"]["s2_resid"]:.5f} vs '
            f'{v["statsmodels"]["s2_resid"]:.5f}, max relative difference {v["max_rel_diff_variance"]:.1e}, max |beta diff| '
            f'{v["max_abs_beta_diff"]:.1e}, REML criterion own {v["own_crit_at_own_optimum"]:.4f} vs at statsmodels estimate '
            f'{v["own_crit_at_statsmodels_estimate"]:.4f}, warnings: {", ".join(v["warnings"]) or "none"}' for c, v in xc.items()) + '.']
    L += ['', 'Reading notes (generated text, no new numbers): the split of the fixed construction variance between S, M and S x M '
          'depends on the coding of the unbalanced design (expert at 1.0 mm only), and so, slightly, does the within-heart construction '
          'share (corr v_endo 21.7% with the task114 S x M column dropped instead, 22.0% here; login-node check, 20 resamples); the '
          'random-effect variance components and fitted values do not. Fixed-term variances are biased upward in resamples '
          '(estimation noise in b adds variance), so some point estimates lie below their percentile interval.', '']
    return L


def write(R, path):
    L = ['# I11 full-arm analysis: estimands (generated by code/full/analysis/run_analysis.py)', '',
         f'Runtime {R["runtime_s"]:.0f} s; bootstrap {R["n_boot"]} resamples over hearts, seed {R["seed"]}. '
         'Definitions: docstring of run_analysis.py and README.md. Values are fractions shown as %. '
         'Interpercentile range = 2.5th to 97.5th percentile. Source CSVs are named in each table.', '']
    E1 = R['E1_primary']
    L += ['## E1. Primary estimand: between-construction CV of theta_hat/theta*', '',
          f'Source: `{E1["source_csv"]}` (include_primary, corr loss). n = {E1["n_hearts"]} hearts, '
          f'{E1["n_constructions"]} constructions, {E1["n_estimates"]} estimates ({E1["n_censored"]} censored, kept at the bound).', '',
          '| parameter | median CV over hearts [95% CI] | per-heart CV median (2.5th to 97.5th) | median SD of theta_hat/theta* [95% CI] |',
          '|---|---|---|---|']
    for p in ['v_endo', 'v_t']:
        d = E1[p]; ph = d['per_heart']
        L.append(f'| {p} | {pc(d["cv_median"])} {pci(d["ci"])} | {pc(ph["median"])} ({pc(ph["p2_5"])} to {pc(ph["p97_5"])}) | '
                 f'{f3(d["sd_median"])} [{f3(d["sd_ci"][0])}, {f3(d["sd_ci"][1])}] |')
    c = R['E1_per_theta_mean_check']
    L += ['', f'Check (mean instead of median over theta* within heart): v_endo {pc(c["v_endo"])}, v_t {pc(c["v_t"])}.', '']
    rm = R['E1_reference_membership']
    L += [f'Reference construction (correction 2026-10-28): {rm["note"]} Reference rows: theta_hat/theta* median '
          + ', '.join(f'{p} {rm["reference_ratio"][p]["median"]:.3f} (2.5th to 97.5th {rm["reference_ratio"][p]["p2_5"]:.3f} to '
                      f'{rm["reference_ratio"][p]["p97_5"]:.3f})' for p in ['v_endo', 'v_t'])
          + '. Without it: ' + ', '.join(f'{p} {pc(rm["no_ref"][p]["cv_median"])} {pci(rm["no_ref"][p]["ci"])}, difference '
                                        f'{pc(rm["no_ref"][p]["diff_vs"]["diff"])} {pci(rm["no_ref"][p]["diff_vs"]["ci"])}'
                                        for p in ['v_endo', 'v_t']) + '.', '']
    A4 = R['root_flag_A4']['by_resolution']
    L += [f'Primary constructions {R["n_primary_constructions"]}. Root flag (> 10 mm). A4 denominator read literally (non-reference '
          f'constructions of the same resolution, every analysis set, mapped root; MANIFEST): 1.5 mm '
          f'{A4["1.5"]["A4_literal"]["flagged"]} of {A4["1.5"]["A4_literal"]["denominator"]}, 1.0 mm '
          f'{A4["1.0"]["A4_literal"]["flagged"]} of {A4["1.0"]["A4_literal"]["denominator"]}. Primary set only (non-reference '
          f'primary constructions): 1.5 mm {R["root_flag"]["1.5"]["flagged"]} of {R["root_flag"]["1.5"]["denominator"]}, '
          f'1.0 mm {R["root_flag"]["1.0"]["flagged"]} of {R["root_flag"]["1.0"]["denominator"]}. Reconciliation: mesh2 counted RV wall '
          f'3 mm only ({A4["1.5"]["mesh2_rv3_only"]["flagged"]} of {A4["1.5"]["mesh2_rv3_only"]["denominator"]}, '
          f'{A4["1.0"]["mesh2_rv3_only"]["flagged"]} of {A4["1.0"]["mesh2_rv3_only"]["denominator"]}); calibC counted calibrated '
          f'constructions ({A4["1.5"]["calibC_calibrated"]["flagged"]} of {A4["1.5"]["calibC_calibrated"]["denominator"]}, '
          f'{A4["1.0"]["calibC_calibrated"]["flagged"]} of {A4["1.0"]["calibC_calibrated"]["denominator"]}); mapped but not calibrated '
          f'(G1 mask not usable): ' + ', '.join(A4["1.5"]["mapped_not_calibrated"] + A4["1.0"]["mapped_not_calibrated"]) + '.', '',
          'Excluded primary-source constructions (A1 rule; `data/processed/acdc_mesh_v2/MANIFEST.tsv`):', '',
          '| heart | source | resolution mm | reason |', '|---|---|---|---|']
    for x in R['exclusions_primary']:
        L.append(f'| {x["heart"]} | {x["source"]} | {x["resolution_mm"]} | {x["exclusion_reason"]} |')
    L += ['', '## E2. Variance shares of log(theta_hat/theta*) (LMM, REML, random heart intercept)', '',
          'Share = term variance / (segmentation + mesh + SxM + theta* + heart + residual); within-heart construction share = '
          '(S + M + SxM) / (S + M + SxM + residual); segmentation of construction = S / (S + M + SxM). Source `'
          + R['inputs']['armC'] + '`.', '',
          'Label correction 2026-10-28: reference rows have log y close to 0 by construction, so the random heart intercept '
          'absorbs each heart\'s mean offset of its non-reference constructions from the reference. The term reported as '
          '"heart" is therefore the heart-level mean construction offset plus between-heart variation, not between-heart '
          'variation alone (values unchanged; JSON key still "heart"). Within-heart decomposition on non-reference rows: E2c. '
          'Candidate model for a later decision (not fitted): ' + R['E2c_within_heart_nonref']['candidate_model_for_user'] + '.', '']
    chk = R.get('E2_statsmodels_check', {})
    if 'v_t' in chk:
        L += ['Cross-check against statsmodels MixedLM (primary): ' + '; '.join(
            f'{p}: s2_heart {chk[p]["own_s2_heart"]:.5f} vs {chk[p]["statsmodels_s2_heart"]:.5f}, s2_resid {chk[p]["own_s2_resid"]:.5f} vs '
            f'{chk[p]["statsmodels_s2_resid"]:.5f}, max |beta diff| {chk[p]["max_abs_beta_diff"]:.1e}' for p in ['v_endo', 'v_t']) + '.', '']
        L += ['statsmodels warnings (cross-check fit only; the shares come from the own REML fit): ' + '; '.join(
            f'{p}: ' + (', '.join(chk[p]['warnings']) or 'none') for p in ['v_endo', 'v_t']) + '. ' + chk['v_endo']['warning_note'] + '.', '']
    terms = ['segmentation_S', 'mesh_M', 'SxM', 'theta_star', 'heart', 'residual', 'within_heart_construction', 'segmentation_of_construction']
    for k, blk in R['E2_shares'].items():
        b0 = blk['v_t']
        L += [f'### Set: {k} (n = {b0["n_hearts"]} hearts, {b0["n_constructions"]} constructions, {b0["n_rows"]} estimates; '
              f'bootstrap fits ok {b0["n_boot_ok"]}, flagged {b0["n_boot_flag"]})', '',
              '| term | v_endo share [95% CI] | v_t share [95% CI] |', '|---|---|---|']
        for t in terms:
            L.append(f'| {TLAB.get(t, t)} | ' + ' | '.join(f'{pc(blk[p]["shares"][t]["share"])} {pci(blk[p]["shares"][t]["ci"])}' for p in ['v_endo', 'v_t']) + ' |')
        L.append('')
    L += ['### E2b. Per-heart two-way sums-of-squares shares (gates-preview definition; descriptive)', '',
          'Per heart: SS of log(theta_hat/theta*) after removing the theta* mean, split into source, resolution, interaction and the '
          'remainder (theta* x construction); median over hearts [bootstrap 95% CI]. Per-heart values in tables/.', '',
          '| set | term | v_endo | v_t |', '|---|---|---|---|']
    for k, blk in R['E2b_ss_per_heart'].items():
        for t in ['segmentation_S', 'mesh_M', 'SxM', 'residual_theta']:
            L.append(f'| {k} | {t} | ' + ' | '.join(f'{pc(blk[p][t]["median"])} {pci(blk[p][t]["ci"])}' for p in ['v_endo', 'v_t']) + ' |')
    L.append('')
    e2c = R['E2c_within_heart_nonref']
    L += ['### E2c. ' + e2c['label'], '',
          'Per heart: log(theta_hat/theta*) minus its theta* mean, sequential sums of squares in the order source, resolution, '
          'source x resolution, remainder (theta* x construction), as shares of the heart total. Sequential because removing the '
          'reference leaves the expert at 1.0 mm only. Median [25th, 75th percentile] over hearts. Per-heart values: '
          '`tables/ss_typeI_per_heart_nonref_*.csv`.', '',
          '| set | n hearts | term | v_endo | v_t |', '|---|---|---|---|---|']
    for k, blk in e2c['sets'].items():
        for t in ['segmentation_S', 'mesh_M', 'SxM', 'residual_theta']:
            L.append(f'| {blk["description"]} | {blk["n_hearts"]} | {t} | ' + ' | '.join(
                f'{pc(blk[p][t]["median"])} [{pc(blk[p][t]["p25"])}, {pc(blk[p][t]["p75"])}]' for p in ['v_endo', 'v_t']) + ' |')
    L.append('')
    if 'E2d_secondary_heart_source' in R:
        L += e2d_lines(R['E2d_secondary_heart_source'])
    L += ['## E3. Break-even level per comparator (arm N against arm C)', '',
          'Median over hearts of SD_noise against median over hearts of SD_cons; log-linear interpolation in severity; '
          'CI listed low to high in native units. "beyond grid: < 10 dB" or "> 20" means construction SD exceeds the median noise SD '
          'even at the most severe tested level; "> 30 dB" or "< 5" means noise SD exceeds it at the least severe level. '
          f'Sources `{R["inputs"]["armN"]}`, `{R["inputs"]["armC"]}`.', '']
    for k, blk in R['E3_breakeven'].items():
        L += [f'### Set: {k}', '', '| comparator | parameter | unit | SD_cons | SD_noise by level | break-even [95% CI] | '
              'per-heart: in grid / beyond more severe / beyond less severe | boot beyond (more / less) | n hearts |',
              '|---|---|---|---|---|---|---|---|---|']
        for d in blk.values():
            u = d['unit']
            sdn = ', '.join(f'{l}: {v:.4f}' for l, v in d['sd_noise_median'].items())
            L.append(f'| {d["comparator"]} | {d["parameter"]} | {u} | {d["sd_cons_median"]:.4f} | {sdn} | {lev(d["breakeven"], u)} '
                     f'[{lev(d["ci"][0], u)}, {lev(d["ci"][1], u)}] | {d["per_heart_interp"]} / {d["per_heart_beyond_most_severe"]} / '
                     f'{d["per_heart_beyond_least_severe"]} | {d["boot_beyond_most_severe"]} / {d["boot_beyond_least_severe"]} | {d["n_hearts"]} |')
        L.append('')
    X3 = R['E3_extrapolated_DESCRIPTIVE']
    L += ['### E3x. ' + X3['label'], '',
          '| set | comparator | parameter | unit | tested grid | slope of log SD per unit severity | extrapolated level [95% CI] | '
          'inside grid | finite bootstrap | n hearts |', '|---|---|---|---|---|---|---|---|---|---|']
    for k in ['primary', 'amp_loss']:
        for d in X3[k].values():
            L.append(f'| {k} | {d["comparator"]} | {d["parameter"]} | {d["unit"]} | {d["grid"][0]} to {d["grid"][-1]} | '
                     f'{d["slope_logSD_per_severity"]:.3f} | {lev(d["level_extrapolated"])} [{lev(d["ci"][0])}, {lev(d["ci"][1])}] '
                     f'(extrapolated) | {d["within_grid"]} | {d["n_boot_finite"]} | {d["n_hearts"]} |')
    L.append('')
    L += ['## E4. Three-length rule (pre-registered free-rule benchmark)', '',
          f'Source `{R["inputs"]["armC"]}` and `{R["inputs"]["armC_cons"]}`; primary non-reference constructions.', '',
          '| loss | n constructions | n hearts | v_endo R2 [95% CI] | v_endo LOHO R2 [95% CI] | v_t R2 [95% CI] | v_t LOHO R2 [95% CI] | '
          'in-sample >= 0.9 both | LOHO >= 0.9 both |', '|---|---|---|---|---|---|---|---|---|']
    for loss, g in R['E4_three_length'].items():
        cells = []
        for p in ['v_endo', 'v_t']:
            for k in ['R2', 'R2_LOHO']:
                cells.append(f'{g[p][k]["value"]:.3f} [{g[p][k]["ci"][0]:.3f}, {g[p][k]["ci"][1]:.3f}]')
        L.append(f'| {loss} | {g["n_constructions"]} | {g["n_hearts"]} | ' + ' | '.join(cells) +
                 f' | {g["flag_in_sample_R2_ge_0.9_both"]} | {g["flag_LOHO_R2_ge_0.9_both"]} |')
    L += ['', '## E5. Arm T torso bias and arm F model-form bias, beside the construction spread', '',
          f'Comparison (correlation loss; `tables/factor_comparison.csv`):', '',
          '| factor | parameter | metric | value [95% CI] | n |', '|---|---|---|---|---|']
    for d in R['E5_comparison']:
        v = d['value']; ci = d['ci']
        s = f'{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]' if 'SD' in d['metric'] else f'{pc(v)} {pci(ci)}'
        L.append(f'| {d["factor"]} | {d["parameter"]} | {d["metric"]} | {s} | {d["n"]} |')
    L += ['', f'### Arm T (source `{R["inputs"]["armT"]}`, round 2 (per-arm refinement, corrected 2026-10-28); per heart mean over theta*, mean over hearts). '
          f'Max |difference| from the owner\'s analysis.json point values: {R["E5_armT_max_abs_diff_vs_owner"]:.1e}.', '',
          '| loss | ref | registration | set | n hearts | v_endo signed | v_endo abs | v_t signed | v_t abs | bound hits | fit loss median [IQR] |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    order = {'trans': 0, 'icp': 1}
    for d in sorted(R['E5_armT'], key=lambda d: (d['loss'] != 'corr', d['ref'], order[d['reg']], d['set'])):
        L.append(f'| {d["loss"]} | {d["ref"]} | {d["reg"]} | {d["set"]} | {d["n_hearts"]} | ' +
                 ' | '.join(f'{pc(d[k][0])} {pci(d[k][1])}' for k in ['v_endo_signed', 'v_endo_abs', 'v_t_signed', 'v_t_abs']) +
                 f' | {d["bound_hits"]}/{d["n_estimates"]} | {d["fval_median"]:.3f} [{d["fval_iqr"][0]:.3f}, {d["fval_iqr"][1]:.3f}] |')
    L += ['', f'### Arm F (source `{R["inputs"]["armF"]}`; geometric mean over hearts of theta_hat/theta*; set re = reaction-eikonal '
          'targets, tmpl = pure-eikonal control, re_dt12p5 = half time step). Owner primary values: '
          f'v_endo {R["E5_armF_owner_primary"]["v_endo"]}, v_t {R["E5_armF_owner_primary"]["v_t"]}.', '',
          '| set | mesh mm | loss | window | n hearts | censored | v_endo GM [t 95% CI] [boot 95% CI] | v_t GM [t 95% CI] [boot 95% CI] | fit loss median |',
          '|---|---|---|---|---|---|---|---|---|']
    for d in sorted(R['E5_armF'], key=lambda d: (d['set'] != 're', d['vox'] != 1.5, d['loss'] != 'corr', d['window'] != 'qrs')):
        cells = []
        for p in ['v_endo', 'v_t']:
            x = d[p]
            cells.append(f'{x["gm"]:.3f} [{f3(x["t_ci"][0])}, {f3(x["t_ci"][1])}] [{f3(x["boot_ci"][0])}, {f3(x["boot_ci"][1])}]')
        L.append(f'| {d["set"]} | {d["vox"]} | {d["loss"]} | {d["window"]} | {d["n_hearts"]} | {d["n_censored"]}/{d["n_estimates"]} | '
                 + ' | '.join(cells) + f' | {d["fval_median"]:.4f} |')
    g6 = R['E6_group']
    L += ['', f'## E6. EXPLORATORY: per-heart CV by ACDC group (associations only; Holm over {g6["family_size"]} tests)', '',
          f'Source `{R["inputs"]["armC"]}` (primary set).', '', '| group | v_endo CV median (2.5th to 97.5th) | v_t CV median (2.5th to 97.5th) | n |',
          '|---|---|---|---|']
    for g, d in g6['group_cv'].items():
        L.append(f'| {g} | {pc(d["v_endo"]["median"])} ({pc(d["v_endo"]["p2_5"])} to {pc(d["v_endo"]["p97_5"])}) | '
                 f'{pc(d["v_t"]["median"])} ({pc(d["v_t"]["p2_5"])} to {pc(d["v_t"]["p97_5"])}) | {d["v_t"]["n"]} |')
    L += ['', '| test | parameter | effect | estimate [95% CI] | statistic | P | P Holm | n |', '|---|---|---|---|---|---|---|---|']
    for t in g6['tests']:
        kw = t['effect'] == 'eta^2_H'
        est = f'{t["estimate"]:.3f} [{t["ci"][0]:.3f}, {t["ci"][1]:.3f}]' if kw else f'{pc(t["estimate"])} {pci(t["ci"])}'
        pf = (lambda x: '< 0.001' if x < 0.001 else f'{x:.3f}' if x < 0.01 else f'{x:.2f}')
        eff = f'eta^2_H = {t["effect_formula"]}' if kw else t['effect']
        L.append(f'| {t["test"]} | {t["parameter"]} | {eff} | {est} | {t["statistic"]} | {pf(t["p"])} | {pf(t["p_holm"])} | {t["n"]} |')
        if kw:
            e = t['epsilon2_R']
            L.append(f'| {t["test"]} | {t["parameter"]} | epsilon^2_R = {e["formula"]} | {e["estimate"]:.3f} [{e["ci"][0]:.3f}, '
                     f'{e["ci"][1]:.3f}] | same test | | | {t["n"]} |')
    L += ['', 'Label correction 2026-10-28: the Kruskal-Wallis effect size was labelled epsilon^2; the formula used, (H - k + 1)/(n - k), '
          'is eta^2_H. Values unchanged; epsilon^2_R = H/(n - 1) added from the same bootstrap resamples.']
    L += ['', '## E7. Sensitivities of the primary estimand', '',
          'Difference = median CV of the set minus median CV of the comparison set, on the hearts both contain, hearts resampled jointly.', '',
          '| set | compared with | n hearts | n constructions | censored | v_endo CV [95% CI] | v_endo difference [95% CI] | v_t CV [95% CI] | v_t difference [95% CI] |',
          '|---|---|---|---|---|---|---|---|---|']
    for d in R['E7_sensitivity_cv']:
        L.append(f'| {d["label"]} | {d["vs"]} | {d["n_hearts"]} | {d["n_constructions"]} | {d["n_censored"]}/{d["n_estimates"]} | ' +
                 ' | '.join(f'{pc(d[p]["cv_median"])} {pci(d[p]["ci"])} | {pc(d[p]["diff_vs"]["diff"])} {pci(d[p]["diff_vs"]["ci"])}'
                            for p in ['v_endo', 'v_t']) + ' |')
    b3 = R.get('E7_B3_rv2_lower_bracket')
    if b3:
        L.append(f'| {b3["label"]} | {b3["vs"]} | {b3["n_hearts"]} | {b3["n_constructions"]} | {b3["n_censored"]}/{b3["n_estimates"]} | ' +
                 ' | '.join(f'{pc(b3[p]["cv_median"])} {pci(b3[p]["ci"])} | {pc(b3[p]["diff_vs"]["diff"])} {pci(b3[p]["diff_vs"]["ci"])}'
                            for p in ['v_endo', 'v_t']) + ' |')
        u = b3['upper_bracket_same_resolution']
        L += ['', f'Amendment 2 B3 (added after full-arm data, no new runs): the RV wall 2 mm level at 1.5 mm is not estimable '
              f'({b3["n_1.5mm_fail_G2_F3"]} of {b3["n_1.5mm_constructions"]} constructions fail G2 F3), so the 1.0 mm RV wall 2 mm set '
              f'({b3["n_hearts"]} hearts) is the stated substitute for the lower bracket; not pre-registered. Bracket at 1.0 mm against '
              f'RV wall 3 mm on the same {b3["n_hearts"]} hearts: 2 mm v_t ' + f'{pc(b3["v_t"]["diff_vs"]["diff"])} {pci(b3["v_t"]["diff_vs"]["ci"])}'
              f', 4 mm v_t {pc(u["v_t"]["diff_vs"]["diff"])} {pci(u["v_t"]["diff_vs"]["ci"])}; v_endo 2 mm '
              f'{pc(b3["v_endo"]["diff_vs"]["diff"])} {pci(b3["v_endo"]["diff_vs"]["ci"])}, 4 mm {pc(u["v_endo"]["diff_vs"]["diff"])} '
              f'{pci(u["v_endo"]["diff_vs"]["ci"])}. The row repeats the values of the "RV wall 2 mm, 1.0 mm only" row above under the B3 label.']
    L += ['', 'Trio per heart (CV, median over theta*):', '', '| set | heart | v_endo | v_t |', '|---|---|---|---|']
    for d in R['E7_sensitivity_cv']:
        if 'per_heart' in d:
            for h, v in d['per_heart'].items():
                L.append(f'| {d["label"]} | {h} | {pc(v["v_endo"])} | {pc(v["v_t"])} |')
    x = R['E7_rv2_at_1.5mm']
    L += ['', f'RV wall 2 mm at 1.5 mm: {x["n_constructions"]} construction(s) ({", ".join(x["constructions"])}); {x["note"]}.', '',
          '## Fit quality beside the estimates (fval = emulator loss at theta_hat)', '',
          '| set | n | censored | fval median (2.5th to 97.5th) | censored by source |', '|---|---|---|---|---|']
    for d in R['fit_quality']:
        bs = '; '.join(f'{s} {v["censored"]}/{v["n"]}' for s, v in d['by_source'].items())
        L.append(f'| {d["set"]} | {d["n_estimates"]} | {d["n_censored"]} | {d["fval"]["median"]:.3f} ({d["fval"]["p2_5"]:.3f} to '
                 f'{d["fval"]["p97_5"]:.3f}) | {bs} |')
    if 'figures' in R:
        L += ['', '## Draft figures', '']
        for f in R['figures']:
            L.append(f'- `{f["pdf"]}`, `{f["png"]}`, raw data `{f["csv"]}`; figqa defects: {f["qa"]}')
    L += ['', '## Correction 2026-10-28', '',
          'Follow-up round; no pre-registered definition changed and every previously reported number is unchanged (checked '
          'against the results.json of job 5395517). Old value, new value, reason:', '',
          '| item | old | new | reason |', '|---|---|---|---|',
          '| E2 term "heart" | label "heart" (random heart intercept) | label "heart-level mean construction offset plus between-heart '
          'variation"; values unchanged; E2c added | reference rows are fixed near log y = 0, so the intercept absorbs each heart\'s mean '
          'construction offset |',
          '| E6 effect size | label "epsilon^2", (H - k + 1)/(n - k) | label "eta^2_H" (same values); epsilon^2_R = H/(n - 1) added | '
          'formula is eta^2_H |',
          f'| root flag, A4 denominator | 1.5 mm {R["root_flag"]["1.5"]["flagged"]} of {R["root_flag"]["1.5"]["denominator"]}, 1.0 mm '
          f'{R["root_flag"]["1.0"]["flagged"]} of {R["root_flag"]["1.0"]["denominator"]} (primary set, presented as the A4 count) | '
          f'A4 literal 1.5 mm {A4["1.5"]["A4_literal"]["flagged"]} of {A4["1.5"]["A4_literal"]["denominator"]}, 1.0 mm '
          f'{A4["1.0"]["A4_literal"]["flagged"]} of {A4["1.0"]["A4_literal"]["denominator"]}; primary-set count kept beside it | '
          'A4 says non-reference constructions of the same resolution, not primary only |',
          '| E1 reference construction | included, membership not stated | stated as a member of S x M (STUDY_PLAN 3 and 4.1); '
          'no-reference sensitivity added | its ratio is about 1 by construction |',
          '| statsmodels warning | printed once, fit not named | recorded per parameter (E2 cross-check) | report, do not hide |',
          '| E3 | beyond grid only | unchanged; extrapolated descriptive E3x added, labelled | user request; not pre-registered |']
    if 'E2d_secondary_heart_source' in R:
        L += ['', '## Addition (Amendment 2)', '',
              'Added after the full-arm data, as declared in `final/AMENDMENT_2_2026-10-28.md` before these analyses were run. No '
              'pre-registered estimand, line or primary value changed; every earlier numeric value of results.json is unchanged. '
              'B2: section E2d (secondary model with a heart x source variance component; E2 stays the pre-registered model). '
              'B3: labelled line in the E7 table. ' + ('B1 and B4: sections B1 and B4 below (integration owner).' if 'B4_E3_extended_grid' in R else 'B1 and B4 are reported by their own owners and are not in this file.')]
    if 'B1_smooth_layer_SENSITIVITY' in R:
        L += b1_lines(R['B1_smooth_layer_SENSITIVITY'])
    if 'B4_E3_extended_grid' in R:
        L += b4_lines(R['B4_E3_extended_grid'])
    open(path, 'w').write('\n'.join(L) + '\n')


def _pp(x, d=1):
    """difference in percentage points"""
    return 'nan' if x is None or x != x else f'{100 * x:+.{d}f}'


def b1_lines(D):
    if 'skipped' in D:
        return ['', '## B1. Smooth-surface fast layer (Amendment 2 B1)', '', f'Skipped: {D["skipped"]}.']
    qc, ut = D['qc'], D['unit_test']
    L = ['', '## B1. Smooth-surface fast layer (Amendment 2 B1, declared sensitivity, added after the full-arm data)', '',
         f'Source `{D["inputs"]["estimates_sdf"]}` (arm C rerun with the B1 layer, round 2): {qc["n_hearts"]} hearts, '
         f'{qc["n_constructions"]} primary constructions, {qc["n_rows"]} estimates, {qc["n_nan_ratio"]} NaN, keys unmatched with the A2 '
         f'primary rows {qc["n_keys_unmatched"]}, maximum |theta* difference| {qc["max_abs_theta_star_diff"]:.1e}. Censored: corr '
         f'{qc["n_censored"]["corr"]} (A2 {qc["n_censored_A2"]["corr"]}), amp {qc["n_censored"]["amp"]} (A2 {qc["n_censored_A2"]["amp"]}); '
         f'kept at the bound. The primary estimand stays on the A2 layer.', '',
         'E1 CV of theta_hat/theta*, median over hearts [95% CI]. Difference = median B1 CV minus median A2 CV, hearts resampled '
         'jointly (pre-declared form, as E7); median of per-heart differences beside it (descriptive). Differences in percentage points.', '',
         '| loss | parameter | B1 CV [95% CI] | A2 CV [95% CI] | B1 minus A2 [95% CI] | median per-heart difference [95% CI] | n hearts |',
         '|---|---|---|---|---|---|---|']
    for l_ in ['corr', 'amp']:
        d = D['E1'][l_]
        for p in ['v_endo', 'v_t']:
            x = d[p]
            L.append(f'| {l_} | {p} | {pc(x["cv_median"])} {pci(x["ci"])} | {pc(x["A2"]["cv_median"])} {pci(x["A2"]["ci"])} | '
                     f'{_pp(x["diff_vs_A2"]["diff"])} [{_pp(x["diff_vs_A2"]["ci"][0])}, {_pp(x["diff_vs_A2"]["ci"][1])}] | '
                     f'{_pp(x["median_per_heart_diff"]["diff"])} [{_pp(x["median_per_heart_diff"]["ci"][0])}, '
                     f'{_pp(x["median_per_heart_diff"]["ci"][1])}] | {x["diff_vs_A2"]["n_common"]} |')
    L += ['', 'Layer share of myocardial volume, median (2.5th to 97.5th percentile) over constructions:', '',
          '| resolution | n | B1 | A2 | B1/A2 | A2 only | B1 only, maximum |', '|---|---|---|---|---|---|---|']
    for r_, x in D['layer_share'].items():
        L.append(f'| {r_ if r_ == "both" else r_ + " mm"} | {x["n_constructions"]} | {pc(x["B1"]["median"])} ({pc(x["B1"]["p2_5"])} to '
                 f'{pc(x["B1"]["p97_5"])}) | {pc(x["A2"]["median"])} ({pc(x["A2"]["p2_5"])} to {pc(x["A2"]["p97_5"])}) | '
                 f'{x["ratio_B1_over_A2"]["median"]:.3f} ({x["ratio_B1_over_A2"]["p2_5"]:.3f} to {x["ratio_B1_over_A2"]["p97_5"]:.3f}) | '
                 f'{pc(x["A2_only"]["median"])} | {x["B1_only"]["max"]:.1e} |')
    L += ['', f'Unit test against its lines ({ut["lines"]}): result **{ut["result"]}** (owner JSON PASS = {ut["owner_json_PASS"]}).', '',
          '| voxel | poses within +-15% | B1 error median (min to max) | A2 error median |', '|---|---|---|---|']
    for v, x in ut['by_resolution'].items():
        L.append(f'| {v} mm | {x["n_within_15pct"]} of {x["n_poses"]} | {pc(x["err_B1_median"])} ({pc(x["err_B1_min"])} to '
                 f'{pc(x["err_B1_max"])}) | {pc(x["err_A2_median"])} |')
    L += ['', f'Median volume ratio 1.0/1.5 mm: {ut["ratio_1p0_over_1p5_recomputed"]:.3f} (owner {ut["ratio_1p0_over_1p5"]:.3f}); '
          f'line 0.85 to 1.15. Other cases ({ut["note"]}): ' +
          '; '.join(f'{o["case"]} {o["vox_mm"]:.1f} mm B1 {pc(o["err_B1"])}, A2 {pc(o["err_A2"])}' for o in ut['other_cases']) + '.']
    return L


def b4_lines(D):
    if 'skipped' in D:
        return ['', '## B4. E3 break-even, extended grid (Amendment 2 B4)', '', f'Skipped: {D["skipped"]}.']
    qc = D['qc']
    L = ['', '## B4. E3 break-even, ' + D['label'], '', 'Rule: ' + D['rule'], '',
         'Grid: ' + '; '.join(f'{c} {v}' for c, v in D['grid'].items()) +
         f'. Inputs `{D["inputs"]["armN"]}` + `{D["inputs"]["armN_ext"]}` ({qc["n_rows_A3"]:,} + {qc["n_rows_ext"]:,} rows; header '
         f'identical {qc["header_identical"]}; duplicate keys {qc["n_duplicate_keys"]}; NaN in the extension {qc["n_nan_ratio_ext"]}). '
         f'Self-check, generalised block on the A3 grid reproduces E3: {qc["selfcheck_reproduces_E3_primary"]}.', '']
    for k_, lab in [('primary', 'correlation loss (primary)'), ('amp_loss', 'amplitude loss (secondary)'),
                    ('censored_dropped', 'correlation loss, censored replicates dropped (sensitivity)')]:
        L += [f'### B4, {lab}', '',
              '| comparator | parameter | median SD_cons | break-even [95% CI] | resamples beyond most / least severe | hearts interpolated / beyond most / beyond least |',
              '|---|---|---|---|---|---|']
        for key, x in D[k_].items():
            L.append(f'| {x["comparator"]} ({x["unit"]}) | {x["parameter"]} | {x["sd_cons_median"]:.4f} | {lev(x["breakeven"])} '
                     f'[{lev(x["ci"][0])}, {lev(x["ci"][1])}] | {x["boot_beyond_most_severe"]} / {x["boot_beyond_least_severe"]} | '
                     f'{x["per_heart_interp"]} / {x["per_heart_beyond_most_severe"]} / {x["per_heart_beyond_least_severe"]} |')
        L.append('')
        L += ['Median noise SD over hearts by level (' + lab + '):', '', '| comparator | parameter | level: median SD_noise |', '|---|---|---|']
        for key, x in D[k_].items():
            L.append(f'| {x["comparator"]} | {x["parameter"]} | ' + ', '.join(f'{l}: {v:.4f}' for l, v in x['sd_noise_median'].items()) + ' |')
        L.append('')
    C = D['comparison_with_E3x']
    L += ['### B4, measured break-even against the E3x extrapolation', '', C['label'] + '.', '',
          '| loss | comparator | parameter | E3, A3 grid | measured, extended grid [95% CI] | E3x [95% CI] | measured minus E3x [95% CI] | resamples used |',
          '|---|---|---|---|---|---|---|---|']
    for x in C['rows']:
        L.append(f'| {x["loss"]} | {x["comparator"]} ({x["unit"]}) | {x["parameter"]} | {lev(x["E3_A3_grid"])} | {lev(x["measured_extended"])} '
                 f'[{lev(x["measured_ci"][0])}, {lev(x["measured_ci"][1])}] | {lev(x["E3x_extrapolated"])} [{lev(x["E3x_ci"][0])}, '
                 f'{lev(x["E3x_ci"][1])}] | {x["diff_measured_minus_E3x"]:+.1f} [{x["diff_ci"][0]:+.1f}, {x["diff_ci"][1]:+.1f}] | '
                 f'{x["n_boot_diff_finite"]} |')
    L += ['', 'Censored noise replicates at the B4 levels (estimate at a prior bound, kept at the bound; noise SD at heavily '
          'censored levels is a lower bound): ' + '; '.join(f'{r_["loss"]} {r_["comparator"]} {r_["level"]}: {r_["censored"]}'
                                                          for r_ in D['noise_censored_ext']) + '.']
    return L
