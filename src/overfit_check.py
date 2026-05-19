import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, permutation_test_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import openpyxl
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
wb = openpyxl.load_workbook(
    r"C:\Users\nechy\Downloads\BioImage_Project\eosine\CA_Out\Combined_statistics_ALLfibFeatures2.xlsx",
    read_only=True)
ws = wb['CAcombined']
rows = list(ws.iter_rows(values_only=True))
df = pd.DataFrame(rows[1:], columns=rows[0])
wb.close()

for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['pat'] = df['image label'].str.extract(r'eosine(TCGA-[A-Z0-9]+-[A-Z0-9]+)_\d+')

pat_mean = df.groupby('pat')[['Omni Test', 'angle median']].mean()
pat_std  = df.groupby('pat')[['Omni Test', 'angle median']].std()
pat_mean.columns = ['Omni_mean', 'angle_median_mean']
pat_std.columns  = ['Omni_std',  'angle_median_std']
pat_df = pd.concat([pat_mean, pat_std], axis=1).reset_index()

clinical = pd.read_csv(
    r'C:\Users\nechy\Downloads\BioImage_Project\data_clinical_patient.txt',
    sep='\t', comment='#', header=0)
clinical['OS_MONTHS'] = pd.to_numeric(clinical['OS_MONTHS'], errors='coerce')
clinical['AGE']       = pd.to_numeric(clinical['AGE'], errors='coerce')
clinical['SEX_NUM']   = clinical['SEX'].map({'Male': 0, 'Female': 1})

merged = pat_df.merge(clinical[['PATIENT_ID', 'OS_MONTHS', 'AGE', 'SEX_NUM']],
                      left_on='pat', right_on='PATIENT_ID')
median_os = merged['OS_MONTHS'].median()
merged['label'] = (merged['OS_MONTHS'] > median_os).astype(int)
y = merged['label'].values

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                   'Omni_std', 'angle_median_std',
                                   'AGE', 'SEX_NUM']].values)

# ============================================================
# PERMUTATION TEST — balanced_accuracy
# ============================================================
pipe = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', SVC(probability=True, random_state=42,
                kernel='rbf', C=0.1, class_weight='balanced'))
])

print("Running permutation test (100 permutations)...")
print("This may take a few minutes...\n")

score, perm_scores, p_value = permutation_test_score(
    pipe, X, y,
    scoring='balanced_accuracy',
    cv=LeaveOneOut(),
    n_permutations=100,
    random_state=42,
    n_jobs=-1
)

real_auc = 0.722

print(f"Real balanced_accuracy: {score:.3f}")
print(f"Permutation mean:       {perm_scores.mean():.3f}")
print(f"Permutation std:        {perm_scores.std():.3f}")
print(f"p-value:                {p_value:.3f}")
print()

if p_value < 0.05:
    print(" Signal is SIGNIFICANT (p < 0.05) — model found real pattern!")
elif p_value < 0.10:
    print(" Signal is BORDERLINE (p < 0.10) — weak but possible signal")
else:
    print(" Signal is NOT significant (p > 0.05) — possible overfitting")

# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(perm_scores, bins=20, color='#1B7FCC', alpha=0.7,
        edgecolor='white', label=f'Permutation scores (n=100)')
ax.axvline(perm_scores.mean(), color='gray', linewidth=1.5, linestyle=':',
           label=f'Permutation mean = {perm_scores.mean():.3f}')
ax.axvline(score, color='red', linewidth=2.5, linestyle='--',
           label=f'Real score = {score:.3f}  (AUC={real_auc:.3f})')

ax.set_xlabel('Balanced Accuracy', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(f'Permutation Test — Collagen + Clinical\n'
             f'p-value = {p_value:.3f} | Real AUC = {real_auc:.3f} | n=100',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\permutation_test.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: permutation_test.png")