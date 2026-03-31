import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('test_results.csv', comment='#', header=None,
                 names=['ar_h', 'ar_l', 'ai_h', 'ai_l', 'br_h', 'br_l', 'bi_h', 'bi_l',
       'ref_re_h', 'ref_re_l', 'ref_im_h', 'ref_im_l', 'rel_err_norm', 'rel_err_fast', 'K'])

rel_err_norm = df['rel_err_norm'].values
rel_err_fast = df['rel_err_fast'].values
K = df['K'].values

worst_norm = np.max(rel_err_norm)
worst_fast = np.max(rel_err_fast)

with open('worst_rel_error.txt', 'w') as f:
    f.write(f"Worst rel_err_norm: {worst_norm:.6e}\n")
    f.write(f"Worst rel_err_fast: {worst_fast:.6e}\n")

#---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(rel_err_norm, bins=50, alpha=0.5, label='rel_err_norm', density=True)
ax.hist(rel_err_fast, bins=50, alpha=0.5, label='rel_err_fast', density=True)
ax.set_xlabel('Relative Error')
ax.set_ylabel('Density')
ax.set_yscale('log')
ax.legend()
ax.set_title('Distribution of Relative Errors')
plt.tight_layout()
plt.savefig('rel_error_distribution.png', dpi=150)
plt.close()

#---------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(K, rel_err_norm, alpha=0.3, s=1, label='rel_err_norm')
ax2.scatter(K, rel_err_fast, alpha=0.3, s=1, label='rel_err_fast')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Conditioning Number (K)')
ax2.set_ylabel('Relative Error')
ax2.legend()
ax2.set_title('Relative Error vs Conditioning Number')
plt.tight_layout()
plt.savefig('rel_error_vs_K.png', dpi=150)
plt.close()

#---------------------------------------------------------------------------------

sig_digits_norm = -np.floor(np.minimum(np.log10(rel_err_norm + 1e-300),0))
sig_digits_fast = -np.floor(np.minimum(np.log10(rel_err_fast + 1e-300),0))

logK = np.log10(K)
logK_bins = np.floor(logK).astype(int)
unique_K_bins = np.unique(logK_bins)

sig_digits_norm_by_K = [np.min(sig_digits_norm[logK_bins == k]) for k in unique_K_bins]
sig_digits_fast_by_K = [np.min(sig_digits_fast[logK_bins == k]) for k in unique_K_bins]

K_centers = 10.0**unique_K_bins

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(K_centers, sig_digits_norm_by_K, 'o-', label='rel_err_norm', markersize=4)
ax3.plot(K_centers, sig_digits_fast_by_K, 's-', label='rel_err_fast', markersize=4)
ax3.set_xscale('log')
ax3.set_xlabel('Conditioning Number (K)')
ax3.set_ylabel('Significant Digits')
ax3.legend()
ax3.set_title('Significant Digits vs Conditioning Number')
plt.tight_layout()
plt.savefig('sig_digits_vs_K.png', dpi=150)
plt.close()

#---------------------------------------------------------------------------------

n_bins = 100
log_edges = np.logspace(np.log10(K.min()), np.log10(K.max()), n_bins + 1)
bin_idx = np.clip(np.digitize(K, log_edges) - 1, 0, n_bins - 1)
bin_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

max_norm  = np.full(n_bins, np.nan)
max_fast  = np.full(n_bins, np.nan)
mean_norm = np.full(n_bins, np.nan)
mean_fast = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = bin_idx == i
    if mask.any():
        max_norm[i]  = rel_err_norm[mask].max()
        max_fast[i]  = rel_err_fast[mask].max()
        mean_norm[i] = rel_err_norm[mask].mean()
        mean_fast[i] = rel_err_fast[mask].mean()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(bin_centers, max_norm,  color='steelblue',  linewidth=1.2, label='max norm')
ax2.plot(bin_centers, mean_norm, color='steelblue',  linewidth=1.2, linestyle='--', label='mean norm')
ax2.plot(bin_centers, max_fast,  color='darkorange', linewidth=1.2, label='max fast')
ax2.plot(bin_centers, mean_fast, color='darkorange', linewidth=1.2, linestyle='--', label='mean fast')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Conditioning Number (K)')
ax2.set_ylabel('Relative Error')
ax2.legend()
ax2.set_title('Relative Error (max and mean) vs Conditioning Number')
plt.tight_layout()
plt.savefig('rel_error_vs_K_binned.png', dpi=150)
plt.close()

#---------------------------------------------------------------------------------

def plot_input_vs_error(df, err_col):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    inputs = [('ar_h', 'a Re (high)'), ('ai_h', 'a Im (high)'),
              ('br_h', 'b Re (high)'), ('bi_h', 'b Im (high)')]
    for ax, (col, label) in zip(axes.flat, inputs):
        ax.scatter(df[col], df[err_col], alpha=0.3, s=1)
        ax.set_xlabel(label)
        ax.set_ylabel('Relative Error')
        ax.set_yscale('log')
    fig.suptitle(f'{err_col} vs Input Magnitude', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{err_col}_vs_input.png', dpi=150)
    plt.close()

plot_input_vs_error(df, 'rel_err_norm')
plot_input_vs_error(df, 'rel_err_fast')

#---------------------------------------------------------------------------------

u2 = 2**(-2*53)
y_norm = df['rel_err_norm'] / df['K'] / u2
y_fast = df['rel_err_fast'] / df['K'] / u2

max_norm = np.full(n_bins, np.nan)
max_fast = np.full(n_bins, np.nan)
mean_norm = np.full(n_bins, np.nan)
mean_fast = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = bin_idx == i
    if mask.any():
        max_norm[i] = y_norm.values[mask].max()
        max_fast[i] = y_fast.values[mask].max()
        mean_norm[i] = y_norm.values[mask].mean()
        mean_fast[i] = y_fast.values[mask].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, max_vals, mean_vals, label in zip(axes,
        [max_norm, max_fast], [mean_norm, mean_fast], ['norm', 'fast']):
    ax.plot(bin_centers, max_vals, color='steelblue', linewidth=1.2, label=f'max {label}')
    ax.plot(bin_centers, mean_vals, color='steelblue', linewidth=1.2, linestyle='--', label=f'mean {label}')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_xlabel('K')
    ax.set_ylabel('Relative Error / K / u^2')
    ax.legend()
plt.suptitle(f'u^2 = 2^(-2*53) = {u2:.2e}')
plt.tight_layout()
plt.savefig('rel_err_over_K_over_u2.png', dpi=150)
plt.close()
