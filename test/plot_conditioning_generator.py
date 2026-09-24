# Plots for res_cond/generator_samples.csv (from exec/test_conditioning_generator):
#   res_cond/branch_distribution.png  which branch i generate_abcd_mp draws and which one produces the pair
#   res_cond/K_coverage.png           how well the achieved K fill [K_MIN, K_MAX], vs number of samples
#   res_cond/attempts.png             attempts needed per generated pair

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'res_cond/generator_samples.csv'
BRANCH_LABELS = ['0: q>0, p+q>0', '1: q>0, p+q<0', '2: q<0, p+q>0', '3: q<0, p+q<0']

with open(CSV_PATH) as f:
    header = f.readline()
m = re.search(r'\[([0-9.e+]+), ([0-9.e+]+)\], max_tries=(\d+)', header)
K_min, K_max, max_tries = float(m.group(1)), float(m.group(2)), int(m.group(3))
log_lo, log_hi = np.log10(K_min), np.log10(K_max)

df = pd.read_csv(CSV_PATH, comment='#')
ok = df[df['branch'] >= 0]
n_total, n_ok = len(df), len(ok)
title_suffix = f'{n_total:,} requested pairs, {n_total - n_ok:,} failed after {max_tries:,} attempts'
decade = np.floor(np.log10(ok['K_dw'])).astype(int)

#-------------------------------------------------------------------------------
# Branch distribution.

tried = df[['tried0', 'tried1', 'tried2', 'tried3']].sum().values
accepted = np.array([(ok['branch'] == i).sum() for i in range(4)])
acceptance = accepted / tried

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 1.6]})
x = np.arange(4)
w = 0.38
bars_t = ax.bar(x - w / 2, tried / tried.sum(), w, color='lightsteelblue', label='drawn (all attempts)')
bars_a = ax.bar(x + w / 2, accepted / accepted.sum(), w, color='steelblue', label='produced the pair')
for bars, vals in ((bars_t, tried / tried.sum()), (bars_a, accepted / accepted.sum())):
    for bar, v in zip(bars, vals):
        ax.annotate(f'{v:.4f}', (bar.get_x() + bar.get_width() / 2, v), ha='center', va='bottom', fontsize=9)
for i in range(4):
    ax.annotate(f'accept {acceptance[i]:.3f}', (x[i], 0.02), ha='center', fontsize=9, color='white')
ax.set_xticks(x)
ax.set_xticklabels(BRANCH_LABELS, fontsize=9)
ax.set_ylabel('Share')
ax.set_ylim(0, 0.35)
ax.legend()
ax.set_title('Branch share (acceptance = produced / drawn)')

decades = np.arange(int(log_lo), int(np.ceil(log_hi)) + 1)
bottom = np.zeros(len(decades))
for i in range(4):
    share = np.array([((decade == d) & (ok['branch'] == i)).sum() / max((decade == d).sum(), 1) for d in decades])
    ax2.bar(decades, share, bottom=bottom, label=BRANCH_LABELS[i])
    bottom += share
ax2.axhline(0.25, color='k', lw=0.8, ls=':')
ax2.axhline(0.50, color='k', lw=0.8, ls=':')
ax2.axhline(0.75, color='k', lw=0.8, ls=':')
ax2.set_xlabel('log10(K) of the generated pair (decade)')
ax2.set_ylabel('Share of pairs in that decade')
ax2.set_title('Producing branch per decade of K (dotted = uniform)')
ax2.legend(fontsize=8, loc='upper right')
fig.suptitle(f'generate_abcd_mp: branch i\n{title_suffix}')
plt.tight_layout()
plt.savefig('res_cond/branch_distribution.png', dpi=150)
plt.close(fig)

#-------------------------------------------------------------------------------
# K coverage.

logK = np.log10(ok['K_dw'].values)
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

ax = axes[0]
counts = np.array([(decade == d).sum() for d in decades])
ax.bar(decades, counts, color='steelblue')
expected = n_ok / (log_hi - log_lo)
ax.axhline(expected, color='darkorange', ls='--', label=f'uniform in log10(K): {expected:,.0f} per decade')
ax.set_xlabel('decade of K (log10)')
ax.set_ylabel('pairs')
ax.set_title('Pairs per decade of K')
ax.legend(fontsize=9)

# Gaps between consecutive achieved log10(K), using the first N pairs in generation order,
# including the uncovered ends [log_lo, min] and [max, log_hi]. Points that double-word rounding
# pushed outside [log_lo, log_hi] (a few just above K_MAX) are outside the space being measured.
ax = axes[1]
logK_in = logK[(logK >= log_lo) & (logK <= log_hi)]
n_out = len(logK) - len(logK_in)
Ns = np.unique(np.append(np.logspace(1, np.log10(len(logK_in)), 40).astype(int), len(logK_in)))
mean_gap, max_gap = [], []
for N in Ns:
    s = np.sort(logK_in[:N])
    g = np.diff(np.concatenate(([log_lo], s, [log_hi])))
    mean_gap.append(g.mean())
    max_gap.append(g.max())
# Expected max of the N+1 gaps of N uniform points: span*H_{N+1}/(N+1), std ~ span/(N+1)*sqrt(sum 1/k^2).
# See CONDITIONING_GENERATOR.md for the derivation.
span = log_hi - log_lo
k = np.arange(1, Ns[-1] + 2)
H = np.cumsum(1.0 / k)
H2 = np.cumsum(1.0 / k**2)
m = Ns + 1
exp_max = span * H[m - 1] / m
std_max = span * np.sqrt(H2[m - 1]) / m
ax.fill_between(Ns, exp_max - std_max, exp_max + std_max, color='grey', alpha=0.25, label='expected max +-1 sigma (approx.)')
ax.loglog(Ns, exp_max, 'k-', lw=0.8, label='span*H(N+1)/(N+1) (expected max, exact)')
ax.loglog(Ns, span * np.log(Ns) / Ns, 'k:', label='span*ln(N)/N (leading term)')
ax.loglog(Ns, span / (Ns + 1), 'k--', lw=0.8, label='span/(N+1) (mean, exact)')
ax.loglog(Ns, max_gap, 'o-', color='steelblue', ms=3, label='max gap')
ax.loglog(Ns, mean_gap, 's-', color='darkorange', ms=3, label='mean gap')
ax.annotate(f'N={Ns[-1]:,}: max {max_gap[-1]:.2e}, mean {mean_gap[-1]:.2e} decades',
            (0.02, 0.03), xycoords='axes fraction', fontsize=9)
ax.set_xlabel('N = number of pairs requested')
ax.set_ylabel('gap between consecutive K [decades of K]')
ax.set_title(f'Distance to the next achieved K ({n_out} pairs outside the range excluded)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

ax = axes[2]
tgt = ok['target_K'].values
dev_mpfr = np.abs(ok['K_mpfr'].values - tgt) / tgt
dev_dw = np.abs(ok['K_dw'].values - tgt) / tgt
# K values are stored as doubles, so deviations below ~1e-16 read as exactly 0; draw those on the floor.
floor = 1e-17
ax.scatter(tgt, np.maximum(dev_dw, floor), s=1, alpha=0.3, color='darkorange', label='K of DW inputs')
ax.scatter(tgt, np.maximum(dev_mpfr, floor), s=1, alpha=0.3, color='steelblue', label='K of 1024-bit numbers')
ax.plot([K_min, K_max], [K_min * 2.0**-106, K_max * 2.0**-106], 'k:', label='K*u^2')
ax.axhline(floor, color='grey', lw=0.8)
ax.annotate('0 in double precision (below ~1e-16)', (0.02, 0.04), xycoords='axes fraction', fontsize=9, color='grey')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(floor / 3, 1)
ax.set_xlabel('target K')
ax.set_ylabel('|K_achieved - K_target| / K_target')
ax.set_title('How far the achieved K is from the target')
ax.legend(fontsize=8, markerscale=6)

fig.suptitle(f'generate_abcd_mp: coverage of K (target log-uniform in [{K_min:.0e}, {K_max:.0e}])\n{title_suffix}')
plt.tight_layout()
plt.savefig('res_cond/K_coverage.png', dpi=150)
plt.close(fig)

#-------------------------------------------------------------------------------
# Attempts per generated pair.

att = ok['attempts'].values
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
n_max = att.max()
values = np.arange(1, n_max + 1)
observed = np.array([(att == v).sum() for v in values]) / n_ok
ax.bar(values, observed, color='steelblue', label='observed')
p_accept = n_ok / att.sum()
ax.plot(values, p_accept * (1 - p_accept) ** (values - 1), 'o--', color='darkorange', ms=4,
        label=f'geometric, p = {p_accept:.4f} (pairs / attempts)')
ax.set_yscale('log')
ax.set_xlabel('attempts to produce one pair')
ax.set_ylabel('fraction of pairs')
ax.set_title(f'Attempts per pair: mean {att.mean():.3f}, median {np.median(att):.0f}, max {n_max}')
ax.legend()

mean_att = [att[decade == d].mean() if (decade == d).any() else np.nan for d in decades]
max_att = [att[decade == d].max() if (decade == d).any() else np.nan for d in decades]
ax2.plot(decades, max_att, 'o-', color='steelblue', label='max')
ax2.plot(decades, mean_att, 's-', color='darkorange', label='mean')
for d, v in zip(decades, mean_att):
    if not np.isnan(v):
        ax2.annotate(f'{v:.2f}', (d, v), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7)
ax2.set_xlabel('decade of K (log10)')
ax2.set_ylabel('attempts')
ax2.set_title('Attempts per pair vs K')
ax2.legend()
ax2.grid(alpha=0.3)
fig.suptitle(f'generate_abcd_mp: attempts\n{title_suffix}')
plt.tight_layout()
plt.savefig('res_cond/attempts.png', dpi=150)
plt.close(fig)

#-------------------------------------------------------------------------------

print(title_suffix)
for i in range(4):
    print(f'branch {BRANCH_LABELS[i]:16s} drawn {tried[i] / tried.sum():.4f}  produced {accepted[i] / accepted.sum():.4f}  '
          f'acceptance {acceptance[i]:.4f}')
print(f'attempts per pair: mean {att.mean():.3f}, max {n_max}')
print(f'gaps at N={Ns[-1]:,}: mean {mean_gap[-1]:.3e}, max {max_gap[-1]:.3e} decades '
      f'(expected max {exp_max[-1]:.3e} +- {std_max[-1]:.1e}, leading term {span * np.log(Ns[-1]) / Ns[-1]:.3e})')
print(f'|K_dw - target|/target: median {np.median(dev_dw):.2e}, max {dev_dw.max():.2e}; '
      f'|K_mpfr - target|/target: max {dev_mpfr.max():.2e}')
print('Wrote res_cond/branch_distribution.png, res_cond/K_coverage.png, res_cond/attempts.png')
