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

sig_digits_norm = -np.floor(np.log10(rel_err_norm + 1e-300))
sig_digits_fast = -np.floor(np.log10(rel_err_fast + 1e-300))

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
