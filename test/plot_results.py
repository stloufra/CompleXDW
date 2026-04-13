import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('test_results.csv', comment='#', header=None,
                 names=['ar_h', 'ar_l', 'ai_h', 'ai_l', 'br_h', 'br_l', 'bi_h', 'bi_l',
       'ref_re_h', 'ref_re_l', 'ref_im_h', 'ref_im_l', 'rel_err_acc_norm', 'rel_err_acc_un', 'rel_err_sloppy_un', 'K'])

rel_err_acc_norm = df['rel_err_acc_norm'].values
rel_err_acc_un = df['rel_err_acc_un'].values
rel_err_sloppy_un = df['rel_err_sloppy_un'].values
K = df['K'].values

worst_acc_norm = np.max(rel_err_acc_norm)
worst_acc_un = np.max(rel_err_acc_un)
worst_sloppy_un = np.max(rel_err_sloppy_un)

with open('worst_rel_error.txt', 'w') as f:
    f.write(f"Worst rel_err_acc_norm: {worst_acc_norm:.6e}\n")
    f.write(f"Worst rel_err_acc_un: {worst_acc_un:.6e}\n")
    f.write(f"Worst rel_err_sloppy_un: {worst_sloppy_un:.6e}\n")

#-------------------------------------------------------------------------------

avg_acc_norm = np.mean(rel_err_acc_norm)
avg_acc_un = np.mean(rel_err_acc_un)
avg_sloppy_un = np.mean(rel_err_sloppy_un)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(rel_err_acc_norm, bins=50, alpha=0.5, label='rel_err_acc_norm', density=True)
ax.hist(rel_err_acc_un, bins=50, alpha=0.5, label='rel_err_acc_un', density=True)
ax.hist(rel_err_sloppy_un, bins=50, alpha=0.5, label='rel_err_sloppy_un', density=True)
ax.set_xlabel('Relative Error')
ax.set_ylabel('Density')
ax.set_yscale('log')
ax.legend()

textstr = f'Avg rel_err_acc_norm: {avg_acc_norm:.16g}\nAvg rel_err_acc_un: {avg_acc_un:.16g}\nAvg rel_err_sloppy_un: {avg_sloppy_un:.16g}'
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

ax.set_title('Distribution of Relative Errors')
plt.tight_layout()
plt.savefig('rel_error_distribution.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(K, rel_err_acc_norm, alpha=0.3, s=1, label='rel_err_acc_norm')
ax2.scatter(K, rel_err_acc_un, alpha=0.3, s=1, label='rel_err_acc_un')
ax2.scatter(K, rel_err_sloppy_un, alpha=0.3, s=1, label='rel_err_sloppy_un')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Conditioning Number (K)')
ax2.set_ylabel('Relative Error')
ax2.legend()

textstr2 = f'Avg rel_err_acc_norm: {avg_acc_norm:.16g}\nAvg rel_err_acc_un: {avg_acc_un:.16g}\nAvg rel_err_sloppy_un: {avg_sloppy_un:.16g}'
props2 = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax2.text(0.95, 0.05, textstr2, transform=ax2.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props2)

ax2.set_title('Relative Error vs Conditioning Number')
plt.tight_layout()
plt.savefig('rel_error_vs_K.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------

sig_digits_acc_norm = -np.floor(np.minimum(np.log10(rel_err_acc_norm + 1e-300),0))
sig_digits_acc_un = -np.floor(np.minimum(np.log10(rel_err_acc_un + 1e-300),0))
sig_digits_sloppy_un = -np.floor(np.minimum(np.log10(rel_err_sloppy_un + 1e-300),0))

logK = np.log10(K)
logK_bins = np.floor(logK).astype(int)
unique_K_bins = np.unique(logK_bins)

sig_digits_acc_norm_by_K = [np.min(sig_digits_acc_norm[logK_bins == k]) for k in unique_K_bins]
sig_digits_acc_un_by_K = [np.min(sig_digits_acc_un[logK_bins == k]) for k in unique_K_bins]
sig_digits_sloppy_un_by_K = [np.min(sig_digits_sloppy_un[logK_bins == k]) for k in unique_K_bins]

K_centers = 10.0**unique_K_bins

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(K_centers, sig_digits_acc_norm_by_K, 'o-', label='rel_err_acc_norm', markersize=4)
ax3.plot(K_centers, sig_digits_acc_un_by_K, 's-', label='rel_err_acc_un', markersize=4)
ax3.plot(K_centers, sig_digits_sloppy_un_by_K, '^-', label='rel_err_sloppy_un', markersize=4)
ax3.set_xscale('log')
ax3.set_xlabel('Conditioning Number (K)')
ax3.set_ylabel('Significant Digits')
ax3.legend()

textstr3 = f'Avg rel_err_acc_norm: {avg_acc_norm:.16g}\nAvg rel_err_acc_un: {avg_acc_un:.16g}\nAvg rel_err_sloppy_un: {avg_sloppy_un:.16g}'
props3 = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax3.text(0.05, 0.05, textstr3, transform=ax3.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='left', bbox=props3)

ax3.set_title('Significant Digits vs Conditioning Number')
plt.tight_layout()
plt.savefig('sig_digits_vs_K.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------

n_bins = 100
log_edges = np.logspace(np.log10(K.min()), np.log10(K.max()), n_bins + 1)
bin_idx = np.clip(np.digitize(K, log_edges) - 1, 0, n_bins - 1)
bin_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

max_acc_norm  = np.full(n_bins, np.nan)
max_acc_un  = np.full(n_bins, np.nan)
max_sloppy_un  = np.full(n_bins, np.nan)
mean_acc_norm = np.full(n_bins, np.nan)
mean_acc_un = np.full(n_bins, np.nan)
mean_sloppy_un = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = bin_idx == i
    if mask.any():
        max_acc_norm[i]  = rel_err_acc_norm[mask].max()
        max_acc_un[i]  = rel_err_acc_un[mask].max()
        max_sloppy_un[i]  = rel_err_sloppy_un[mask].max()
        mean_acc_norm[i] = rel_err_acc_norm[mask].mean()
        mean_acc_un[i] = rel_err_acc_un[mask].mean()
        mean_sloppy_un[i] = rel_err_sloppy_un[mask].mean()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(bin_centers, max_acc_norm,  color='steelblue',  linewidth=1.2, label='max acc_norm')
ax2.plot(bin_centers, mean_acc_norm, color='steelblue',  linewidth=1.2, linestyle='--', label='mean acc_norm')
ax2.plot(bin_centers, max_acc_un,  color='green', linewidth=1.2, label='max acc_un')
ax2.plot(bin_centers, mean_acc_un, color='green', linewidth=1.2, linestyle='--', label='mean acc_un')
ax2.plot(bin_centers, max_sloppy_un,  color='darkorange', linewidth=1.2, label='max sloppy_un')
ax2.plot(bin_centers, mean_sloppy_un, color='darkorange', linewidth=1.2, linestyle='--', label='mean sloppy_un')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Conditioning Number (K)')
ax2.set_ylabel('Relative Error')
ax2.legend()

textstr_binned = f'Avg rel_err_acc_norm: {avg_acc_norm:.16g}\nAvg rel_err_acc_un: {avg_acc_un:.16g}\nAvg rel_err_sloppy_un: {avg_sloppy_un:.16g}'
props_binned = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax2.text(0.95, 0.05, textstr_binned, transform=ax2.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props_binned)

ax2.set_title('Relative Error (max and mean) vs Conditioning Number')
plt.tight_layout()
plt.savefig('rel_error_vs_K_binned.png', dpi=150)
plt.close()

#-------------------------------------------------------------------------------

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

plot_input_vs_error(df, 'rel_err_acc_norm')
plot_input_vs_error(df, 'rel_err_acc_un')
plot_input_vs_error(df, 'rel_err_sloppy_un')

#-------------------------------------------------------------------------------

u2 = 2**(-2*53)
y_acc_norm = rel_err_acc_norm / K / u2
y_acc_un = rel_err_acc_un / K / u2
y_sloppy_un = rel_err_sloppy_un / K / u2

max_acc_norm = np.full(n_bins, np.nan)
max_acc_un = np.full(n_bins, np.nan)
max_sloppy_un = np.full(n_bins, np.nan)
mean_acc_norm = np.full(n_bins, np.nan)
mean_acc_un = np.full(n_bins, np.nan)
mean_sloppy_un = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = bin_idx == i
    if mask.any():
        max_acc_norm[i] = y_acc_norm[mask].max()
        max_acc_un[i] = y_acc_un[mask].max()
        max_sloppy_un[i] = y_sloppy_un[mask].max()
        mean_acc_norm[i] = y_acc_norm[mask].mean()
        mean_acc_un[i] = y_acc_un[mask].mean()
        mean_sloppy_un[i] = y_sloppy_un[mask].mean()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, max_vals, mean_vals, label in zip(axes,
        [max_acc_norm, max_acc_un, max_sloppy_un], 
        [mean_acc_norm, mean_acc_un, mean_sloppy_un], 
        ['acc_norm', 'acc_un', 'sloppy_un']):
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
