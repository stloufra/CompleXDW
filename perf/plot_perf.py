import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/plots', exist_ok=True)

df = pd.read_csv('results/perf_times.csv')

array_sizes = sorted(df['array_size'].unique())
n_sizes = len(array_sizes)

n_cols = 4
n_rows = (n_sizes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for idx, size in enumerate(array_sizes):
    df_size = df[df['array_size'] == size]
    iterations = df_size['iteration'].values
    
    ax1 = axes[idx]
    ax1.plot(iterations, df_size['time_acc_norm'], 'o-', label='acc_norm', markersize=2, alpha=0.7)
    ax1.plot(iterations, df_size['time_acc_un'], 's-', label='acc_un', markersize=2, alpha=0.7)
    ax1.plot(iterations, df_size['time_sloppy_un'], '^-', label='sloppy_un', markersize=2, alpha=0.7)
    
    mean_norm = df_size['time_acc_norm'].mean()
    mean_un = df_size['time_acc_un'].mean()
    mean_sloppy = df_size['time_sloppy_un'].mean()
    
    ax1.axhline(y=mean_norm, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=mean_un, color='orange', linestyle='--', alpha=0.5)
    ax1.axhline(y=mean_sloppy, color='green', linestyle='--', alpha=0.5)
    
    speedup_un = mean_norm / mean_un
    speedup_sloppy = mean_norm / mean_sloppy
    
    ax1.text(0.98, 0.98, f'norm: {mean_norm:.2f}\nun: {mean_un:.2f}\nsloppy: {mean_sloppy:.2f}\nspeedups: {speedup_un:.2f}x / {speedup_sloppy:.2f}x',
            transform=ax1.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time (ns/op)')
    ax1.set_title(f'Array Size = {size}')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

for idx in range(len(array_sizes), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Performance: Times & Speedups (K=100)')
plt.tight_layout()
plt.savefig('results/plots/perf_times_and_speedup.png', dpi=150)
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

print("Plots saved: results/plots/perf_times_and_speedup.png, results/plots/perf_speedup_by_size.png")