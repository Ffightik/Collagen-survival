import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# === Load ===
merged = pd.read_csv(r'C:\Users\nechy\Downloads\BioImage_Project\results\features_33patients.csv')


top5 = ['Omni Test', 'angle median', 'fiber weight', 'overall orientation', 'angle skewness']
top5_labels = ['Omni Test', 'Angle Median', 'Fiber Weight', 'Overall Orientation', 'Angle Skewness']

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

for i, (f, lab) in enumerate(zip(top5, top5_labels)):
    ax = axes[i]
    s = merged[merged['label']==0][f].dropna().values
    l = merged[merged['label']==1][f].dropna().values
    p = stats.mannwhitneyu(s, l, alternative='two-sided').pvalue

    bp = ax.boxplot([s, l], patch_artist=True,
                    tick_labels=['Short', 'Long'],
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    bp['boxes'][0].set_facecolor('#D95F0255')
    bp['boxes'][0].set_edgecolor('#D95F02')
    bp['boxes'][1].set_facecolor('#1B7FCC55')
    bp['boxes'][1].set_edgecolor('#1B7FCC')

    # Individual points with jitter
    np.random.seed(42)
    ax.scatter(np.ones(len(s)) + np.random.normal(0, 0.05, len(s)), s,
               color='#D95F02', alpha=0.7, s=40, zorder=3)
    ax.scatter(np.ones(len(l))*2 + np.random.normal(0, 0.05, len(l)), l,
               color='#1B7FCC', alpha=0.7, s=40, zorder=3)

    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.set_title(f'{lab}\np = {p:.3f}  {sig}', fontsize=11, fontweight='bold')
    ax.set_ylabel(lab, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('T Features: Short vs Long Survival \nMann-Whitney U test',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\boxplots_top5.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved!")