import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('test_results.csv', comment='#', header=None,
                 names=['ref_re_h', 'ref_re_l', 'ref_im_h', 'ref_im_l', 'rel_err_norm', 'rel_err_fast', 'K'])

rel_err_norm = df['rel_err_norm'].values
rel_err_fast = df['rel_err_fast'].values
K = df['K'].values

worst_norm = np.max(rel_err_norm)
worst_fast = np.max(rel_err_fast)

with open('worst_rel_error.txt', 'w') as f:
    f.write(f"Worst rel_err_norm: {worst_norm:.6e}\n")
    f.write(f"Worst rel_err_fast: {worst_fast:.6e}\n")

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

n_bins = 200
log_edges = np.logspace(np.log10(K.min()), np.log10(K.max()), n_bins + 1)
bin_idx = np.clip(np.digitize(K, log_edges) - 1, 0, n_bins - 1)

bin_centers = np.sqrt(log_edges[:-1] * log_edges[1:])
max_norm = np.full(n_bins, np.nan)
max_fast = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = bin_idx == i
    if mask.any():
        max_norm[i] = rel_err_norm[mask].max()
        max_fast[i] = rel_err_fast[mask].max()

window = 100
def rolling_max(x, w):
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - w // 2)
        hi = min(len(x), i + w // 2 + 1)
        valid = x[lo:hi]
        valid = valid[~np.isnan(valid)]
        if len(valid):
            out[i] = valid.max()
    return out

roll_norm = rolling_max(max_norm, window)
roll_fast = rolling_max(max_fast, window)

valid_norm = ~np.isnan(roll_norm)
valid_fast = ~np.isnan(roll_fast)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(K, rel_err_norm, alpha=0.3, s=1, color='steelblue', label='rel_err_norm')
ax2.scatter(K, rel_err_fast, alpha=0.3, s=1, color='orange', label='rel_err_fast')
ax2.plot(bin_centers[valid_norm], roll_norm[valid_norm], color='steelblue', linewidth=1.5, label='rolling max norm')
ax2.plot(bin_centers[valid_fast], roll_fast[valid_fast], color='darkorange', linewidth=1.5, label='rolling max fast')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Conditioning Number (K)')
ax2.set_ylabel('Relative Error')
ax2.legend()
ax2.set_title('Relative Error max rolling vs Conditioning Number ')
plt.tight_layout()
plt.savefig('rel_error_vs_K_rolling.png', dpi=150)
plt.close()

print(f"Worst rel_err_norm: {worst_norm:.6e}")
print(f"Worst rel_err_fast: {worst_fast:.6e}")
print("Plot saved to rel_error_distribution.png")
print("Plot saved to rel_error_vs_K.png")
print("Worst errors written to worst_rel_error.txt")
