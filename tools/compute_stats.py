"""
Compute statistics for the CTCF paper:
  1) Hodges-Lehmann estimator + 95% CI for paired comparisons
  2) Fold% for OASIS (check availability)
  3) SDlogJ for IXI (check availability)
  4) Epoch training times from available logs
"""

import os, re, sys
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.stats import wilcoxon
from scipy.stats import norm
from math import comb

BASE = r"C:\Users\user\Documents\Education\MasterWork\repos\CTCF"
INFER = os.path.join(BASE, "results", "infer")


def hodges_lehmann_paired(x, y, alpha=0.05):
    """
    For paired observations: d_i = x_i - y_i
    HL = median of all Walsh averages (d_i + d_j)/2 for i <= j
    CI from sorted Walsh averages using Wilcoxon signed-rank distribution.
    """
    d = np.array(x) - np.array(y)
    n = len(d)

    # Walsh averages
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((d[i] + d[j]) / 2.0)
    walsh = np.sort(walsh)
    hl = np.median(walsh)

    # Wilcoxon p-value
    try:
        if n <= 25: stat, pval = wilcoxon(d, alternative='two-sided', method='exact')
        else: stat, pval = wilcoxon(d, alternative='two-sided', method='approx')
    except Exception as e:
        stat, pval = np.nan, np.nan

    # CI from Walsh averages using normal approximation for rank
    # K_alpha = n(n+1)/4 - z_{alpha/2} * sqrt(n(n+1)(2n+1)/24)
    N_walsh = len(walsh)  # = n*(n+1)/2
    z = norm.ppf(1 - alpha / 2)

    # For large samples, use normal approx to Wilcoxon distribution
    mu_T = n * (n + 1) / 4
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    K = int(np.round(mu_T - z * sigma_T))

    if K < 1: K = 1
    if K > N_walsh: K = N_walsh

    # CI: (walsh[K-1], walsh[N_walsh - K])
    ci_lo = walsh[K - 1]
    ci_hi = walsh[N_walsh - K]

    return {
        'hl': hl,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'p_wilcoxon': pval,
        'wilcoxon_stat': stat,
        'n': n,
        'method': 'exact' if n <= 25 else 'approx',
        'K': K,
        'N_walsh': N_walsh,
    }


def load_csv(dataset, model):
    """Load per_case.csv, trying both 'best' and 'best.pth' subdirs."""
    for subdir in ['best', 'best.pth']:
        path = os.path.join(INFER, dataset, model, subdir, 'per_case.csv')
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"No per_case.csv for {dataset}/{model}")


print("=" * 80)
print("TASK 1: Hodges-Lehmann estimator + 95% CI (paired comparisons)")
print("=" * 80)


# OASIS
print("\n### OASIS (N=19)")
oasis_ctcf = load_csv('OASIS', 'ctcf')
oasis_tmdca = load_csv('OASIS', 'tm-dca')
oasis_utsr = load_csv('OASIS', 'utsrmorph')

# Verify pairing by case_id
assert list(oasis_ctcf['case_id']) == list(oasis_tmdca['case_id']), "OASIS case_id mismatch CTCF vs TM-DCA"
assert list(oasis_ctcf['case_id']) == list(oasis_utsr['case_id']), "OASIS case_id mismatch CTCF vs UTSRMorph"

oasis_metrics = ['dice_mean', 'hd95_mean', 'sdlogj']
oasis_comparisons = [
    ('CTCF vs TM-DCA', oasis_ctcf, oasis_tmdca),
    ('CTCF vs UTSRMorph', oasis_ctcf, oasis_utsr),
]

for comp_name, df_a, df_b in oasis_comparisons:
    print(f"\n  {comp_name}:")
    for metric in oasis_metrics:
        # For HD95 and SDlogJ: lower is better, so CTCF - baseline should be negative
        # For Dice: higher is better, so CTCF - baseline should be positive
        result = hodges_lehmann_paired(df_a[metric].values, df_b[metric].values)
        direction = "higher is better" if metric == 'dice_mean' else "lower is better"
        print(f"    {metric} ({direction}):")
        print(f"      HL = {result['hl']:+.6f}  95% CI = [{result['ci_lo']:+.6f}, {result['ci_hi']:+.6f}]")
        print(f"      Wilcoxon p = {result['p_wilcoxon']:.6e} (method={result['method']}, N={result['n']}, T={result['wilcoxon_stat']}, K={result['K']})")


# IXI
print("\n### IXI (N=115)")
ixi_ctcf = load_csv('IXI', 'ctcf')
ixi_tmdca = load_csv('IXI', 'tm-dca')
ixi_utsr = load_csv('IXI', 'utsrmorph')

assert list(ixi_ctcf['case_id']) == list(ixi_tmdca['case_id']), "IXI case_id mismatch CTCF vs TM-DCA"
assert list(ixi_ctcf['case_id']) == list(ixi_utsr['case_id']), "IXI case_id mismatch CTCF vs UTSRMorph"

ixi_metrics = ['dice_mean', 'hd95_mean', 'j_leq0_percent']
ixi_comparisons = [
    ('CTCF vs TM-DCA', ixi_ctcf, ixi_tmdca),
    ('CTCF vs UTSRMorph', ixi_ctcf, ixi_utsr),
]

for comp_name, df_a, df_b in ixi_comparisons:
    print(f"\n  {comp_name}:")
    for metric in ixi_metrics:
        result = hodges_lehmann_paired(df_a[metric].values, df_b[metric].values)
        direction = "higher is better" if metric == 'dice_mean' else "lower is better"
        print(f"    {metric} ({direction}):")
        print(f"      HL = {result['hl']:+.6f}  95% CI = [{result['ci_lo']:+.6f}, {result['ci_hi']:+.6f}]")
        print(f"      Wilcoxon p = {result['p_wilcoxon']:.6e} (method={result['method']}, N={result['n']}, T={result['wilcoxon_stat']}, K={result['K']})")