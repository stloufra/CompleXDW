# Plots results/elementwise_<build>.csv written by bench_elementwise:
#   results/elementwise_<type>.png            median ns/element per variant at REFERENCE_SIZE, per build (AoS);
#                                             black ticks = time the flop count predicts from the reference
#   results/elementwise_layout_<type>.png      AoS (XDW[]) vs SoA (XDWSpan), vec build
#   results/elementwise_size_sweep_<type>.png  median ns/element vs array length (vec build, AoS + SoA reference)
#   results/elementwise_cost_vs_accuracy.png   double, vec build: ns/element vs error / (K u^2) from the
#                                             binned conditioning runs in ../test/res/, if present

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REFERENCE_SIZE = 1024
OPS = ('mul', 'div')
BUILD_COLORS = {'vec': 'steelblue', 'scalar': 'darkorange'}
BASELINES = ('naive', 'std::complex')

paths = sorted(glob.glob('results/elementwise_*.csv'))
if not paths:
    raise SystemExit('No results/elementwise_*.csv found, run bench_elementwise first')
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
if 'layout' not in df:
    df['layout'] = 'AoS'  # results from before the SoA variants

stats = (df.groupby(['build', 'type', 'op', 'layout', 'variant', 'flops', 'divisions', 'n'], sort=False)['ns_per_elem']
           .agg(median='median', min='min').reset_index())
aos = stats[stats['layout'] == 'AoS']
builds = [b for b in BUILD_COLORS if b in stats['build'].unique()]


def tick_label(row):
    cost = f"{row.flops} flops" if row.flops > 0 else 'flops ?'
    if row.divisions:
        cost += f", {row.divisions} div"
    return row.variant.replace('/', '\n') + '\n' + cost


def bars(ax, s, op):
    s = s[s['op'] == op]
    order = s.drop_duplicates('variant')
    x = np.arange(len(order))
    width = 0.8 / len(builds)
    for j, build in enumerate(builds):
        sb = s[s['build'] == build].set_index('variant').loc[order['variant']]
        ref = sb.iloc[0]
        pos = x + (j - (len(builds) - 1) / 2) * width
        rects = ax.bar(pos, sb['median'], width, color=BUILD_COLORS[build], label=f'{build} build (median)')
        for rect, (variant, row) in zip(rects, sb.iterrows()):
            ax.annotate(f"{row['median']:.2f}\nx{row['median'] / ref['median']:.2f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha='center', va='bottom', fontsize=7)
            if variant not in BASELINES:
                expected = ref['median'] * row['flops'] / ref['flops']
                ax.hlines(expected, rect.get_x(), rect.get_x() + rect.get_width(), colors='k', lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([tick_label(r) for r in order.itertuples()], fontsize=7)
    ax.set_ylabel('ns / element')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')


for T in stats['type'].unique():
    s = aos[(aos['type'] == T) & (aos['n'] == REFERENCE_SIZE)]
    fig, axes = plt.subplots(2, 1, figsize=(20, 13), gridspec_kw={'height_ratios': [1, 1.3]})
    for ax, op in zip(axes, OPS):
        bars(ax, s, op)
        ref_name = s[s['op'] == op]['variant'].iloc[0]
        ax.set_title(f'{op}: ratios relative to {ref_name}; black tick = reference time x flops / reference flops',
                     fontsize=10)
    fig.suptitle(f'Element-wise complex {T}, n = {REFERENCE_SIZE} (a, b, c in L1)')
    plt.tight_layout()
    plt.savefig(f'results/elementwise_{T}.png', dpi=150)
    plt.close(fig)

    s = aos[(aos['type'] == T) & (aos['build'] == builds[0])]
    soa = stats[(stats['type'] == T) & (stats['build'] == builds[0]) & (stats['layout'] == 'SoA')]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, op in zip(axes, OPS):
        for variant, sv in s[s['op'] == op].groupby('variant', sort=False):
            ax.plot(sv['n'], sv['median'], 's--' if variant in BASELINES else 'o-', ms=3, lw=1, label=variant)
        ref_name = s[s['op'] == op]['variant'].iloc[0]
        sv = soa[(soa['op'] == op) & (soa['variant'] == ref_name)]
        if len(sv):
            ax.plot(sv['n'], sv['median'], 'k^-.', ms=4, lw=1.5, label=f'SoA {ref_name}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('array length n')
        ax.set_ylabel('median ns / element')
        ax.set_title(op)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'Element-wise complex {T} vs array length ({builds[0]} build)')
    plt.tight_layout()
    plt.savefig(f'results/elementwise_size_sweep_{T}.png', dpi=150)
    plt.close(fig)


def layout_bars(ax, s, op):
    """AoS vs SoA median per DW variant, annotated with the SoA speedup."""
    s = s[(s['op'] == op) & ~s['variant'].isin(BASELINES)]
    order = s[s['layout'] == 'AoS']['variant'].tolist()
    x = np.arange(len(order))
    width = 0.4
    t = {layout: s[s['layout'] == layout].set_index('variant').loc[order]['median'].values for layout in ('AoS', 'SoA')}
    ax.bar(x - width / 2, t['AoS'], width, color='steelblue', label='AoS: XDW[]')
    rects = ax.bar(x + width / 2, t['SoA'], width, color='seagreen', label='SoA: XDWSpan')
    for rect, a, b in zip(rects, t['AoS'], t['SoA']):
        ax.annotate(f'{b:.2f}\nx{a / b:.2f}', (rect.get_x() + rect.get_width() / 2, b), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('/', '\n') for v in order], fontsize=7)
    ax.set_ylabel('median ns / element')
    ax.set_title(f'{op}: SoA time and speedup over AoS')
    ax.set_ylim(0, 1.25 * max(t['AoS'].max(), t['SoA'].max()))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)


for T in stats['type'].unique():
    s = stats[(stats['type'] == T) & (stats['build'] == 'vec') & (stats['n'] == REFERENCE_SIZE)]
    if not (s['layout'] == 'SoA').any():
        continue
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    for ax, op in zip(axes, OPS):
        layout_bars(ax, s, op)
    fig.suptitle(f'Element-wise complex {T}, n = {REFERENCE_SIZE}, vec build: array of XDW vs XDWSpan')
    plt.tight_layout()
    plt.savefig(f'results/elementwise_layout_{T}.png', dpi=150)
    plt.close(fig)


def binned_bounds(op):
    """Per combo: (max, sample-weighted mean) of error / (K u^2) over all K, or None if not run."""
    path = f'../test/res/binned_results_{op}.csv'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        combos = next(l for l in f if l.startswith('# combos:'))
    names = [n for _, n in re.findall(r'(\d+)=(\S+)', combos)]
    args = ['ar_h', 'ar_l', 'ai_h', 'ai_l', 'br_h', 'br_l', 'bi_h', 'bi_l']
    cols = ['bin', 'K_lo', 'K_hi', 'count']
    for i in range(len(names)):
        cols += [f'c{i}_{c}' for c in ('min', 'max', 'mean', 'nmax', 'nmean')] + [f'c{i}_worst_{a}' for a in args]
    b = pd.read_csv(path, comment='#', header=None, names=cols)
    b = b[b['count'] > 0]
    return {n: (b[f'c{i}_nmax'].max(), np.average(b[f'c{i}_nmean'], weights=b['count'])) for i, n in enumerate(names)}


bounds = {op: binned_bounds(op) for op in OPS}
if any(bounds.values()):
    s = aos[(aos['type'] == 'double') & (aos['build'] == 'vec') & (aos['n'] == REFERENCE_SIZE)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, op in zip(axes, OPS):
        if not bounds[op]:
            ax.set_title(f'{op}: no ../test/res/binned_results_{op}.csv')
            continue
        for row in s[s['op'] == op].itertuples():
            if row.variant not in bounds[op]:
                continue
            nmax, nmean = bounds[op][row.variant]
            ax.scatter(row.median, nmax, color='steelblue')
            ax.scatter(row.median, nmean, facecolors='none', edgecolors='darkorange')
            short = row.variant.replace('Unnormalized', 'U').replace('Normalized', 'N')
            ax.annotate(short, (row.median, nmax), textcoords='offset points', xytext=(0, 6),
                        rotation=90, ha='center', va='bottom', fontsize=7)
        ax.scatter([], [], color='steelblue', label='max over all K')
        ax.scatter([], [], facecolors='none', edgecolors='darkorange', label='mean over all K')
        ax.set_ylim(0, 1.4 * max(nmax for nmax, _ in bounds[op].values()))
        ax.set_xlabel(f'median ns / element (double, vec build, n = {REFERENCE_SIZE})')
        ax.set_ylabel('relative error / (K u^2)')
        ax.set_title(op)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Cost vs accuracy (error from the binned conditioning runs)')
    plt.tight_layout()
    plt.savefig('results/elementwise_cost_vs_accuracy.png', dpi=150)
    plt.close(fig)
    print('Wrote results/elementwise_cost_vs_accuracy.png')
else:
    print('No ../test/res/binned_results_{mul,div}.csv, skipped the cost vs accuracy plot')

print('Wrote ' + ', '.join(f'results/elementwise_{T}.png, results/elementwise_layout_{T}.png, '
                           f'results/elementwise_size_sweep_{T}.png' for T in stats['type'].unique()))
