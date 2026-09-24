import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Must match COMBO_NAMES / combo order in test/src/combo_mul.h.
COMBO_NAMES = ['Madd/Normalized', 'Madd/Unnormalized', 'Accurate/Normalized',
               'Accurate/Unnormalized', 'Sloppy/Normalized', 'Sloppy/Unnormalized']
ERR_COLS = [f'rel_err_{i}' for i in range(6)]

df = pd.read_csv('res/test_results.csv', comment='#', header=None,
                 names=['ar_h', 'ar_l', 'ai_h', 'ai_l', 'br_h', 'br_l', 'bi_h', 'bi_l',
                        'ref_re_h', 'ref_re_l', 'ref_im_h', 'ref_im_l', *ERR_COLS, 'K'])

K = df['K'].values
avg_err = {col: df[col].mean() for col in ERR_COLS}
worst_err = {col: df[col].max() for col in ERR_COLS}

with open('res/worst_rel_error.txt', 'w') as f:
    for col, name in zip(ERR_COLS, COMBO_NAMES):
        f.write(f"Worst {name}: {worst_err[col]:.6e}\n")

#-------------------------------------------------------------------------------
# Distribution of relative errors, all 6 combos overlaid.

fig, ax = plt.subplots(figsize=(10, 6))
for col, name in zip(ERR_COLS, COMBO_NAMES):
    ax.hist(df[col], bins=50, alpha=0.4, label=name, density=True)
ax.set_xlabel('Relative Error')
ax.set_ylabel('Density')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.set_title('Distribution of Relative Errors')
plt.tight_layout()
plt.savefig('res/rel_error_distribution.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------
# Plain scatter of relative error vs K, all 6 combos overlaid.

fig, ax = plt.subplots(figsize=(10, 6))
for col, name in zip(ERR_COLS, COMBO_NAMES):
    ax.scatter(K, df[col], alpha=0.3, s=1, label=name)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Conditioning Number (K)')
ax.set_ylabel('Relative Error')
ax.legend(fontsize=8)
ax.set_title('Relative Error vs Conditioning Number')
plt.tight_layout()
plt.savefig('res/rel_error_vs_K.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------
# Relative error vs input magnitude, one 2x2 grid per combo.

def plot_input_vs_error(df, err_col, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    inputs = [('ar_h', 'a Re (high)'), ('ai_h', 'a Im (high)'),
              ('br_h', 'b Re (high)'), ('bi_h', 'b Im (high)')]
    for ax, (col, label) in zip(axes.flat, inputs):
        ax.scatter(df[col], df[err_col], alpha=0.3, s=1)
        ax.set_xlabel(label)
        ax.set_ylabel('Relative Error')
        ax.set_yscale('log')
    fig.suptitle(f'{name} vs Input Magnitude', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'res/{err_col}_vs_input.png', dpi=150)
    plt.close()

for col, name in zip(ERR_COLS, COMBO_NAMES):
    plot_input_vs_error(df, col, name)
