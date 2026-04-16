import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/plots', exist_ok=True)

df = pd.read_csv('results/perf_times.csv')

array_sizes = sorted(df['array_size'].unique())

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, size in enumerate(array_sizes):
    df_size = df[df['array_size'] == size]
    iterations = df_size['iteration'].values
    
    axes[idx].plot(iterations, df_size['time_acc_norm'], 'o-', label='acc_norm', markersize=2, alpha=0.7)
    axes[idx].plot(iterations, df_size['time_acc_un'], 's-', label='acc_un', markersize=2, alpha=0.7)
    axes[idx].plot(iterations, df_size['time_sloppy_un'], '^-', label='sloppy_un', markersize=2, alpha=0.7)
    
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel('Time (ns/op)')
    axes[idx].set_title(f'Array Size = {size}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Performance Measurement Times per Iteration (K=100)')
plt.tight_layout()
plt.savefig('results/plots/perf_times_by_iteration.png', dpi=150)
plt.close()

fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
axes2 = axes2.flatten()

for idx, size in enumerate(array_sizes):
    df_size = df[df['array_size'] == size]
    
    mean_norm = df_size['time_acc_norm'].mean()
    mean_un = df_size['time_acc_un'].mean()
    mean_sloppy = df_size['time_sloppy_un'].mean()
    
    std_norm = df_size['time_acc_norm'].std()
    std_un = df_size['time_acc_un'].std()
    std_sloppy = df_size['time_sloppy_un'].std()
    
    speedup_un = mean_norm / mean_un
    speedup_sloppy = mean_norm / mean_sloppy
    
    operators = ['acc_norm', 'acc_un', 'sloppy_un']
    means = [mean_norm, mean_un, mean_sloppy]
    stds = [std_norm, std_un, std_sloppy]
    speedups = [1.0, speedup_un, speedup_sloppy]
    
    x = np.arange(len(operators))
    width = 0.35
    
    bars = axes2[idx].bar(x, means, width, yerr=stds, capsize=5, label='Mean Time (ns/op)')
    axes2[idx].set_ylabel('Time (ns/op)')
    axes2[idx].set_xticks(x)
    axes2[idx].set_xticklabels(operators)
    axes2[idx].set_title(f'Array Size = {size}')
    axes2[idx].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        axes2[idx].annotate(f'{height:.1f}\n({speedup:.2f}x)',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

plt.suptitle('Mean Time and Speedup vs acc_norm')
plt.tight_layout()
plt.savefig('results/plots/perf_mean_and_speedup.png', dpi=150)
plt.close()

fig3, ax3 = plt.subplots(figsize=(10, 6))

speedup_un_by_size = []
speedup_sloppy_by_size = []
speedup_un_err = []
speedup_sloppy_err = []

for size in array_sizes:
    df_size = df[df['array_size'] == size]
    mean_norm = df_size['time_acc_norm'].mean()
    mean_un = df_size['time_acc_un'].mean()
    mean_sloppy = df_size['time_sloppy_un'].mean()
    std_norm = df_size['time_acc_norm'].std()
    std_un = df_size['time_acc_un'].std()
    std_sloppy = df_size['time_sloppy_un'].std()
    
    speedup_un = mean_norm / mean_un
    speedup_sloppy = mean_norm / mean_sloppy
    rel_err_un = np.sqrt((std_norm/mean_norm)**2 + (std_un/mean_un)**2)
    rel_err_sloppy = np.sqrt((std_norm/mean_norm)**2 + (std_sloppy/mean_sloppy)**2)
    
    speedup_un_by_size.append(speedup_un)
    speedup_sloppy_by_size.append(speedup_sloppy)
    speedup_un_err.append(speedup_un * rel_err_un)
    speedup_sloppy_err.append(speedup_sloppy * rel_err_sloppy)

array_sizes_arr = np.array(array_sizes)
ax3.errorbar(array_sizes_arr, speedup_un_by_size, yerr=speedup_un_err, fmt='s-', 
             label='acc_un', markersize=8, capsize=5, alpha=0.7)
ax3.errorbar(array_sizes_arr, speedup_sloppy_by_size, yerr=speedup_sloppy_err, fmt='^-', 
             label='sloppy_un', markersize=8, capsize=5, alpha=0.7)

ax3.set_xlabel('Array Size')
ax3.set_ylabel('Speedup (x)')
ax3.set_xscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=36/21, color='green', linestyle='--', label='theoretical sloppy (36/21)', alpha=0.7)
ax3.axhline(y=36/30, color='red', linestyle='--', label='theoretical un (36/30)', alpha=0.7)
ax3.set_title('Speedup vs acc_norm by Array Size')

plt.tight_layout()
plt.savefig('results/plots/perf_speedup_by_size.png', dpi=150)
plt.close()

fig4, axes4 = plt.subplots(2, 4, figsize=(20, 10))
axes4 = axes4.flatten()

for idx, size in enumerate(array_sizes):
    df_size = df[df['array_size'] == size]
    iterations = df_size['iteration'].values
    
    tp_norm = df_size['time_acc_norm'] / size
    tp_un = df_size['time_acc_un'] / size
    tp_sloppy = df_size['time_sloppy_un'] / size
    
    mean_norm = tp_norm.mean()
    mean_un = tp_un.mean()
    mean_sloppy = tp_sloppy.mean()
    
    axes4[idx].plot(iterations, tp_norm, 'o-', label='acc_norm', markersize=2, alpha=0.7)
    axes4[idx].plot(iterations, tp_un, 's-', label='acc_un', markersize=2, alpha=0.7)
    axes4[idx].plot(iterations, tp_sloppy, '^-', label='sloppy_un', markersize=2, alpha=0.7)
    
    axes4[idx].axhline(y=mean_norm, color='blue', linestyle='--', alpha=0.5)
    axes4[idx].axhline(y=mean_un, color='orange', linestyle='--', alpha=0.5)
    axes4[idx].axhline(y=mean_sloppy, color='green', linestyle='--', alpha=0.5)
    
    axes4[idx].text(0.98, 0.98, f'norm: {mean_norm:.2f}\nun: {mean_un:.2f}\nsloppy: {mean_sloppy:.2f}',
                   transform=axes4[idx].transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes4[idx].set_xlabel('Iteration')
    axes4[idx].set_ylabel('Time (ns/op per element)')
    axes4[idx].set_title(f'Array Size = {size}')
    axes4[idx].legend(loc='upper left')
    axes4[idx].grid(True, alpha=0.3)

plt.suptitle('Throughput per Single Operation (normalized by array size)')
plt.tight_layout()
plt.savefig('results/plots/perf_throughput_by_iteration.png', dpi=150)
plt.close()

def remove_outliers_iqr(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 0.2 * iqr
    upper = q3 + 0.2 * iqr
    return (data >= lower) & (data <= upper)

size = 50
df_size = df[df['array_size'] == size].reset_index(drop=True)
iterations = df_size['iteration'].values

tp_norm = df_size['time_acc_norm'] / size
tp_un = df_size['time_acc_un'] / size
tp_sloppy = df_size['time_sloppy_un'] / size

mask_norm = remove_outliers_iqr(tp_norm)
mask_un = remove_outliers_iqr(tp_un)
mask_sloppy = remove_outliers_iqr(tp_sloppy)

tp_norm_clean = tp_norm[mask_norm]
tp_un_clean = tp_un[mask_un]
tp_sloppy_clean = tp_sloppy[mask_sloppy]

iterations_norm = iterations[mask_norm.values]
iterations_un = iterations[mask_un.values]
iterations_sloppy = iterations[mask_sloppy.values]

fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(iterations_norm, tp_norm_clean, 'o-', label='acc_norm', markersize=4, alpha=0.7)
ax5.plot(iterations_un, tp_un_clean, 's-', label='acc_un', markersize=4, alpha=0.7)
ax5.plot(iterations_sloppy, tp_sloppy_clean, '^-', label='sloppy_un', markersize=4, alpha=0.7)

mean_norm = tp_norm_clean.mean()
mean_un = tp_un_clean.mean()
mean_sloppy = tp_sloppy_clean.mean()

ax5.axhline(y=mean_norm, color='blue', linestyle='--', alpha=0.5)
ax5.axhline(y=mean_un, color='orange', linestyle='--', alpha=0.5)
ax5.axhline(y=mean_sloppy, color='green', linestyle='--', alpha=0.5)

ax5.text(0.98, 0.98, f'norm: {mean_norm:.2f}\nun: {mean_un:.2f}\nsloppy: {mean_sloppy:.2f}',
        transform=ax5.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax5.set_xlabel('Iteration')
ax5.set_ylabel('Time (ns/op per element)')
ax5.set_title(f'Throughput (no outliers) - Array Size = {size}')
ax5.legend(loc='upper left')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/perf_throughput_no_outliers_50.png', dpi=150)
plt.close()

print("Plots saved: results/plots/perf_times_by_iteration.png, results/plots/perf_mean_and_speedup.png, results/plots/perf_speedup_by_size.png, results/plots/perf_throughput_by_iteration.png, results/plots/perf_throughput_no_outliers_50.png")