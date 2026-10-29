"""DRAFT figures for the I11 full-arm analysis (called by run_analysis.py). 183 mm wide, PDF + 600 dpi PNG,
raw-data CSV beside each figure. Style: code/figures/vendor/figstyle.py (strict font resolution)."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures', 'vendor'))
import figstyle  # noqa: E402
import figqa  # noqa: E402

W = 183.0
PCOL = {'v_endo': '#0072B2', 'v_t': '#D55E00'}
PLAB = {'v_endo': r'$v_\mathrm{endo}$', 'v_t': r'$v_\mathrm{t}$'}
PMRK = {'v_endo': 'o', 'v_t': 's'}
SCOL = {'expert': '#000000', 'nnunet_task027_acdc': '#0072B2', 'nnunet_task114_mnms': '#009E73',
        'nnunet_ukb_sax': '#E69F00', 'cinema_acdc_sax': '#CC79A7'}
SLAB = {'expert': 'Expert', 'nnunet_task027_acdc': 'nnU-Net Task027', 'nnunet_task114_mnms': 'nnU-Net Task114',
        'nnunet_ukb_sax': 'nnU-Net UKB SAX', 'cinema_acdc_sax': 'CineMA'}
RMRK = {1.5: 'o', 1.0: '^'}
GORDER = ['NOR', 'DCM', 'HCM', 'MINF', 'RV']


def _draft(fig):
    fig.suptitle('DRAFT', x=0.995, ha='right', fontsize=7, color='0.45')


def _plain_log(ax, ticks):
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def _save(fig, stem):
    fig.canvas.draw()
    qa = figqa.report(fig)
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=600)
    try:
        figqa.greyscale_and_downscale(stem + '.png', W)
    except Exception as ex:  # report only
        qa = list(qa) + [f'greyscale_and_downscale failed: {ex!r}']
    plt.close(fig)
    return [str(x) for x in qa]


def fig_a(d, rows, cvtab, R):
    stem = os.path.join(d, 'draft_fig_a_construction_cv')
    cv = cvtab.copy()
    cv['gorder'] = cv.group.map({g: i for i, g in enumerate(GORDER)})
    cv = cv.sort_values(['gorder', 'cv_v_t'])
    xpos = {h: i for i, h in enumerate(cv.index)}
    pts = rows[['patient', 'group', 'construction', 'source', 'resolution_mm', 'istar', 'censored', 'ratio_v_endo', 'ratio_v_t']].copy()
    pts['x'] = pts.patient.map(xpos)
    pts.to_csv(stem + '_points.csv', index=False)
    cv.drop(columns='gorder').assign(x=[xpos[h] for h in cv.index]).to_csv(stem + '_cv.csv')
    fig, axs = figstyle.figure_mm(W, 150, nrows=3, sharex=True, layout='constrained',
                                  gridspec_kw=dict(height_ratios=[1, 1, 1]))
    srcs = list(SCOL)
    for ax, p in zip(axs[:2], ['v_endo', 'v_t']):
        for j, s_ in enumerate(srcs):
            for r_ in [1.5, 1.0]:
                g = pts[(pts.source == s_) & (pts.resolution_mm == r_)]
                off = (j - 2) * 0.12 + (0.04 if r_ == 1.0 else -0.04)
                ax.scatter(g.x + off, g[f'ratio_{p}'], s=3, marker=RMRK[r_], facecolor='none' if r_ == 1.0 else SCOL[s_],
                           edgecolor=SCOL[s_], linewidths=0.5, alpha=0.8, zorder=2)
        ax.axhline(1.0, color='0.3', lw=0.5, zorder=1)
        ax.set_yscale('log')
        ax.set_ylabel(f'{PLAB[p]}: ' + r'$\hat\theta/\theta^*$')
    _plain_log(axs[0], [0.6, 0.8, 1, 1.25, 1.5, 2, 2.5])
    _plain_log(axs[1], [0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5])
    for p in ['v_endo', 'v_t']:
        ax = axs[2]
        m = R['E1_primary'][p]
        ax.scatter([xpos[h] for h in cv.index], 100 * cv[f'cv_{p}'], s=6, marker=PMRK[p], color=PCOL[p], zorder=3,
                   label=f'{PLAB[p]}: median {100 * m["cv_median"]:.1f}% [{100 * m["ci"][0]:.1f}, {100 * m["ci"][1]:.1f}]')
        ax.axhline(100 * m['cv_median'], color=PCOL[p], lw=0.8, ls='--' if p == 'v_t' else '-', zorder=2)
        ax.axhspan(100 * m['ci'][0], 100 * m['ci'][1], color=PCOL[p], alpha=0.15, lw=0, zorder=1)
    axs[2].set_yscale('log')
    _plain_log(axs[2], [1, 2, 5, 10, 20, 50])
    axs[2].set_ylabel('Between-construction CV (%)')
    axs[2].legend(loc='upper left', frameon=False, ncol=2)
    # group separators and labels
    bounds = cv.groupby('group').apply(lambda g: (min(xpos[h] for h in g.index), max(xpos[h] for h in g.index)), include_groups=False)
    for ax in axs:
        for g in GORDER[1:]:
            ax.axvline(bounds[g][0] - 0.5, color='0.7', lw=0.5, zorder=0)
    axs[2].set_xticks([(bounds[g][0] + bounds[g][1]) / 2 for g in GORDER])
    axs[2].set_xticklabels([f'{g} (n = {bounds[g][1] - bounds[g][0] + 1})' for g in GORDER])
    axs[2].set_xlabel('Heart (ACDC test set), grouped by ACDC group, ordered by $v_\\mathrm{t}$ CV within group')
    axs[2].set_xlim(-0.8, len(cv) - 0.2)
    h_ = [plt.Line2D([], [], ls='', marker='s', color=SCOL[s_], markersize=3, label=SLAB[s_]) for s_ in srcs]
    h_ += [plt.Line2D([], [], ls='', marker='o', color='0.3', markersize=3, label='1.5 mm (filled)'),
           plt.Line2D([], [], ls='', marker='^', mfc='none', color='0.3', markersize=3, label='1.0 mm (open)')]
    axs[0].legend(handles=h_, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=False)
    figstyle.panel_labels(axs, ['a', 'b', 'c'], dx=-0.06)
    _draft(fig)
    qa = _save(fig, stem)
    return dict(pdf=stem + '.pdf', png=stem + '.png', csv=stem + '_points.csv; ' + stem + '_cv.csv', qa=qa)


def fig_b(d, R):
    stem = os.path.join(d, 'draft_fig_b_variance_shares')
    terms = ['segmentation_S', 'mesh_M', 'SxM', 'theta_star', 'heart', 'residual']
    tlab = {'segmentation_S': 'Segmentation (S)', 'mesh_M': 'Mesh (M)', 'SxM': 'S x M', 'theta_star': r'$\theta^*$ draw',
            'heart': 'Heart: mean offset\n+ between-heart', 'residual': 'Residual', 'residual_theta': r'Remainder ($\theta^*$ x construction)'}
    rows = []
    for k in ['primary', 'amp_loss']:
        for p in ['v_endo', 'v_t']:
            for t in terms:
                x = R['E2_shares'][k][p]['shares'][t]
                rows.append(dict(panel=f'LMM {k}', parameter=p, term=t, share=x['share'], lo=x['ci'][0], hi=x['ci'][1]))
    for p in ['v_endo', 'v_t']:
        for t in ['segmentation_S', 'mesh_M', 'SxM', 'residual_theta']:
            x = R['E2b_ss_per_heart']['primary'][p][t]
            rows.append(dict(panel='per-heart SS primary', parameter=p, term=t, share=x['median'], lo=x['ci'][0], hi=x['ci'][1]))
    df = pd.DataFrame(rows); df.to_csv(stem + '.csv', index=False)
    fig, axs = figstyle.figure_mm(W, 72, ncols=3, layout='constrained')
    titles = {'LMM primary': 'LMM, correlation loss', 'LMM amp_loss': 'LMM, amplitude loss',
              'per-heart SS primary': 'Per-heart SS, correlation loss'}
    for ax, (pan, title) in zip(axs, titles.items()):
        g = df[df.panel == pan]
        tl = list(dict.fromkeys(g.term))
        for j, p in enumerate(['v_endo', 'v_t']):
            gg = g[g.parameter == p].set_index('term').loc[tl]
            y = np.arange(len(tl)) + (j - 0.5) * 0.38
            ax.barh(y, 100 * gg.share, height=0.36, color=PCOL[p], hatch=None if p == 'v_endo' else '////',
                    edgecolor='white' if p == 'v_endo' else PCOL[p], lw=0.5,
                    fill=(p == 'v_endo'), label=PLAB[p])
            ax.errorbar(100 * gg.share, y, xerr=[np.clip(100 * (gg.share - gg.lo), 0, None), np.clip(100 * (gg.hi - gg.share), 0, None)], fmt='none',
                        ecolor='0.15', elinewidth=0.6, capsize=1.2)
        ax.set_yticks(np.arange(len(tl)))
        ax.set_yticklabels([tlab[t] for t in tl])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Share of variance of log ratio (%)')
        ax.set_title(title)
    axs[0].legend(frameon=False, loc='center right')
    figstyle.panel_labels(axs, ['a', 'b', 'c'], dx=-0.55 if False else -0.12)
    _draft(fig)
    qa = _save(fig, stem)
    return dict(pdf=stem + '.pdf', png=stem + '.png', csv=stem + '.csv', qa=qa)


def fig_c(d, R, nsd, cvtab, COMP):
    stem = os.path.join(d, 'draft_fig_c_noise_vs_construction')
    ctitle = {'white': 'White noise', 'lp40': 'Band-limited (40 Hz)', 'wander': 'Baseline wander', 'disp': 'Electrode displacement'}
    xlab = {'white': 'SNR (dB)', 'lp40': 'SNR (dB)', 'wander': 'Wander amplitude (% of QRS peak-to-peak)', 'disp': 'Displacement SD (mm)'}
    rows = []
    fig, axs = figstyle.figure_mm(W, 105, nrows=2, ncols=4, layout='constrained', sharey='row')
    for i, p in enumerate(['v_endo', 'v_t']):
        be = R['E3_breakeven']['primary']
        sdc = R['E1_primary'][p]
        for j, c in enumerate(COMP):
            ax = axs[i, j]
            levels = COMP[c][1]
            sub = nsd[nsd.comparator == c]
            med = [sub[sub.level == L][f'sd_{p}'].median() for L in levels]
            lo = [sub[sub.level == L][f'sd_{p}'].quantile(0.025) for L in levels]
            hi = [sub[sub.level == L][f'sd_{p}'].quantile(0.975) for L in levels]
            for L, m_, a_, b_ in zip(levels, med, lo, hi):
                rows.append(dict(parameter=p, comparator=c, level=L, unit=COMP[c][0], noise_sd_median=m_, noise_sd_p2_5=a_,
                                 noise_sd_p97_5=b_, cons_sd_median=sdc['sd_median'], cons_sd_ci_lo=sdc['sd_ci'][0],
                                 cons_sd_ci_hi=sdc['sd_ci'][1]))
            ax.fill_between(levels, lo, hi, color=PCOL[p], alpha=0.18, lw=0)
            ax.plot(levels, med, marker=PMRK[p], color=PCOL[p], ms=3, lw=1.0, label='Noise SD (median over hearts)')
            ax.axhline(sdc['sd_median'], color='0.1', lw=0.8, ls='--', label='Construction SD (median, 95% CI)')
            ax.axhspan(sdc['sd_ci'][0], sdc['sd_ci'][1], color='0.5', alpha=0.25, lw=0)
            b = be[f'{c}|{p}']
            if isinstance(b['breakeven'], float):
                ax.plot([b['breakeven']], [sdc['sd_median']], marker='D', ms=4, mfc='white', mec='0.1', mew=0.8, ls='')
            ax.set_yscale('log')
            if COMP[c][0] == 'dB SNR':
                ax.invert_xaxis()
            else:
                ax.set_xscale('log')
                ax.set_xticks(levels); ax.set_xticklabels([str(x) for x in levels]); ax.minorticks_off()
            if i == 0:
                ax.set_title(ctitle[c])
            if i == 1:
                ax.set_xlabel(xlab[c])
            if j == 0:
                ax.set_ylabel(f'SD of {PLAB[p]} ' + r'$\hat\theta/\theta^*$')
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='outside lower center', ncol=2, frameon=False)
    figstyle.panel_labels(axs.ravel(), list('abcdefgh'), dx=-0.12)
    _draft(fig)
    pd.DataFrame(rows).to_csv(stem + '.csv', index=False)
    qa = _save(fig, stem)
    return dict(pdf=stem + '.pdf', png=stem + '.png', csv=stem + '.csv', qa=qa)


def make_all(d, rows, cvtab, R, nsd, COMP):
    os.makedirs(d, exist_ok=True)
    fam = figstyle.use_print_style()
    out = [fig_a(d, rows, cvtab, R), fig_b(d, R), fig_c(d, R, nsd, cvtab, COMP)]
    for o in out:
        o['font'] = fam
    return out
