import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('test_results.csv', comment='#', header=None,
                 names=['ref_re_h', 'ref_re_l', 'ref_im_h', 'ref_im_l', 'rel_err_norm', 'rel_err_fast'])

rel_err_norm = df['rel_err_norm'].values
rel_err_fast = df['rel_err_fast'].values

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

print(f"Worst rel_err_norm: {worst_norm:.6e}")
print(f"Worst rel_err_fast: {worst_fast:.6e}")
print("Plot saved to rel_error_distribution.png")
print("Worst errors written to worst_rel_error.txt")
