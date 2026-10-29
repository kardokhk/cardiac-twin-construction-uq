import json,sys
m=json.load(open('mine.json'))
t=json.load(open('/project/home/p201509/work/cvri/simulation/results/2026-10-29_v03_analyses/results.json'))
rows=[]
def c(name,mine,theirs,tol=1e-9):
    ok=abs(mine-theirs)<=tol*max(1,abs(theirs))
    rows.append((name,'REPRODUCED' if ok else 'NOT',mine,theirs)); 
c('E1 v_t',m['E1_v_t'],t['E1_regression']['v_t']); c('E1 v_endo',m['E1_v_endo'],t['E1_regression']['v_endo'])
for r in t['item1_per_network']:
    s=r['source']
    c(f'{s} n_hearts',m[f'{s}_n'],r['n_hearts']); c(f'{s} n_est',m[f'{s}_nest'],r['n_estimates'])
    if s!='expert_1.0':
        c(f'{s} myo Dice used',m[f'{s}_myo_used'],r['myo_dice_median_used_hearts']); c(f'{s} myo Dice all',m[f'{s}_myo_all'],r['myo_dice_median_all50'])
        c(f'{s} rv Dice used',m[f'{s}_rv_used'],r['rv_dice_median_used_hearts'])
    for p in ['v_t','v_endo']:
        for i,q in enumerate(['median','q25','q75']): c(f'{s} ratio_{p} {q}',m[f'{s}_ratio_{p}'][i],r[f'ratio_{p}_{q}'])
        for i,q in enumerate(['median','ci_lo','ci_hi']):
            c(f'{s} err_{p} {q}',m[f'{s}_e_{p}'][i],r[f'abs_err_{p}_{q}'])
            c(f'{s} alt_err_{p} {q}',m[f'{s}_a_{p}'][i],r[f'alt_abs_err_med_{p}_{q}'])
for kind,tk in [('e','err'),('a','abserr_med')]:
  for p in ['v_t','v_endo']:
    a=m[f'item2_{kind}_{p}']; b=t['item2_rank_concordance'][f'{tk}_{p}']
    c(f'item2 {tk}_{p} n',a['n'],b['n_hearts_complete'])
    for k,v in b['first_counts'].items(): c(f'item2 {tk}_{p} first {k}',a['first'][k],v)
    c(f'item2 {tk}_{p} W',a['W'],b['kendall_w']); c(f'item2 {tk}_{p} W ci lo',a['Wci'][0],b['kendall_w_ci'][0]); c(f'item2 {tk}_{p} W ci hi',a['Wci'][1],b['kendall_w_ci'][1])
    c(f'item2 {tk}_{p} chi2',a['chi2'],b['friedman_chi2']); c(f'item2 {tk}_{p} Friedman P',a['friedman_p'],b['friedman_p'])
    c(f'item2 {tk}_{p} perm P',a['perm_p'],b['permutation_p']); c(f'item2 {tk}_{p} null mean',a['null_mean'],b['w_null_mean'])
    rows.append((f'item2 {tk}_{p} perm P alt scheme / null mean','INFO',a['perm_p_alt'],a['null_mean_alt']))
for lab,tl in [('p4','primary4'),('p4u','primary4_plus_ukbb')]:
  for d in ['myo','rv']:
    for p in ['v_t','v_endo']:
      a=m[f'item3_{lab}_{d}_{p}']; b=t['item3_dice_vs_error'][f'{tl}__{d}_dice__{p}']
      n=f'item3 {tl} {d} {p}'
      c(n+' rho',a['rho'],b['pooled_spearman']); c(n+' ci lo',a['ci'][0],b['pooled_ci'][0]); c(n+' ci hi',a['ci'][1],b['pooled_ci'][1])
      c(n+' npairs',a['npairs'],b['n_pairs']); c(n+' rho alt',a['rho_alt'],b['pooled_spearman_alt_abserr_med'])
      c(n+' within',a['wh'],b['within_heart_mean_spearman']); c(n+' within ci lo',a['whci'][0],b['within_heart_ci'][0]); c(n+' within ci hi',a['whci'][1],b['within_heart_ci'][1])
      c(n+' npos',a['npos'],b['within_heart_n_positive']); c(n+' nneg',a['nneg'],b['within_heart_n_negative'])
for p in ['v_t','v_endo']:
    a=m[f'item4_{p}']; b=t['item4_ensemble_size'][f'complete_hearts_{p}']
    for k in '234':
        c(f'item4 {p} k{k}',a[k]['v'],b['by_k'][k]['value']); c(f'item4 {p} k{k} ci lo',a[k]['ci'][0],b['by_k'][k]['ci'][0]); c(f'item4 {p} k{k} ci hi',a[k]['ci'][1],b['by_k'][k]['ci'][1])
        c(f'item4 {p} k{k} ratio',a[k]['ratio'],b['by_k'][k]['ratio_to_k4']); c(f'item4 {p} k{k} ratio ci lo',a[k]['rci'][0],b['by_k'][k]['ratio_ci'][0]); c(f'item4 {p} k{k} ratio ci hi',a[k]['rci'][1],b['by_k'][k]['ratio_ci'][1])
        rows.append((f'item4 {p} k{k} alt median-over-hearts of mean-over-subsets','INFO',a[k]['alt_median_of_mean'],None))
        c(f'item4 {p} allhearts k{k}',a['allhearts'][k],t['item4_ensemble_size'][f'all_hearts_available_{p}']['value'][k])
    h=a['half']; c(f'item4 {p} half frac',h['frac'],b['half_spread_mean_fraction_of_pairs']); c(f'item4 {p} half ci lo',h['ci'][0],b['half_spread_ci'][0]); c(f'item4 {p} half ci hi',h['ci'][1],b['half_spread_ci'][1])
    c(f'item4 {p} half all',h['all'],b['half_spread_hearts_all_pairs']); c(f'item4 {p} half any',h['any'],b['half_spread_hearts_any_pair'])
c('item4 excluded', sum(m['item4_excluded']), sum(t['item4_excluded_from_complete']))
for lab,tl in [('prim','primary'),('noref','primary_no_ref')]:
  for p in ['v_t','v_endo']:
    a=m[f'item5_{lab}_{p}']; b=t['item5_physical_units'][f'{tl}_{p}']
    c(f'item5 {tl} {p} SD',a[0],b['sd_m_per_s_median']); c(f'item5 {tl} {p} ci lo',a[1],b['ci'][0]); c(f'item5 {tl} {p} ci hi',a[2],b['ci'][1])
    c(f'item5 {tl} {p} iqr lo',a[3],b['per_heart_iqr'][0]); c(f'item5 {tl} {p} iqr hi',a[4],b['per_heart_iqr'][1])
c('item5 true v_t',m['item5_true'][0],t['item5_physical_units']['true_v_t']['median_over_theta_star']); c('item5 true v_endo',m['item5_true'][1],t['item5_physical_units']['true_v_endo']['median_over_theta_star'])
for res in ['1.5','1.0']:
    a=m[f'item6_{res}']; b=t['item6_resolution'][res]
    c(f'item6 {res} n',a['n'],b['n_constructions']); c(f'item6 {res} nodes',a['nodes'],b['node_median']); c(f'item6 {res} tets',a['tets'],b['tet_median'])
    c(f'item6 {res} wall',a['wall'],b['grid_run_wall_s_median_over_constructions']); c(f'item6 {res} ref wall',a['ref_wall'],b['refinement_run_wall_s_median']); c(f'item6 {res} nref',a['nref'],b['n_refinement_runs'])
b=t['item6_resolution']['paired_ratio_1.0_over_1.5']; a=m['item6_ratio']
c('item6 ratio nodes',a['nodes'],b['nodes_median']); c('item6 ratio tets',a['tets'],b['tets_median']); c('item6 ratio wall',a['wall'],b['grid_wall_median']); c('item6 pairs',a['n'],b['n_pairs'])
rows.append(('item6 nodes MANIFEST vs constructions.csv max abs diff','INFO',m['item6_nodes_manifest_vs_constructions_maxabsdiff'],None))
json.dump(rows,open('rows.json','w'))
bad=[r for r in rows if r[1]=='NOT']; print(len(rows),'checks', len(bad),'NOT'); 
for r in bad+[r for r in rows if r[1]=='INFO']: print(r)
