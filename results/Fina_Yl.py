import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import openpyxl

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
    r"C:\Users\nechy\Downloads\BioImage_Project\data_clinical_patient.txt",
    sep='\t', comment='#', header=0)
clinical['OS_MONTHS'] = pd.to_numeric(clinical['OS_MONTHS'], errors='coerce')
clinical['AGE']       = pd.to_numeric(clinical['AGE'], errors='coerce')
clinical['SEX_NUM']   = clinical['SEX'].map({'Male':0,'Female':1})

merged = pat_df.merge(clinical[['PATIENT_ID','OS_MONTHS','AGE','SEX_NUM']],
                      left_on='pat', right_on='PATIENT_ID')
median_os = merged['OS_MONTHS'].median()
merged['label'] = (merged['OS_MONTHS'] > median_os).astype(int)
y = merged['label'].values

print(f"Patients: {len(merged)}, Median OS: {median_os:.1f} months")
print(f"Short=0: {(y==0).sum()}, Long=1: {(y==1).sum()}")

imputer = SimpleImputer(strategy='median')

# ============================================================
# ONLY Collagen + Clinical
# ============================================================
X = imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                   'Omni_std',  'angle_median_std',
                                   'AGE', 'SEX_NUM']].values)

# ============================================================
# NESTED LOOCV — SVM
# ============================================================
param_grid = {
    'clf__kernel':       ['linear', 'rbf'],
    'clf__C':            [0.01, 0.1, 0.5, 1.0],
    'clf__class_weight': ['balanced', None]
}

loo = LeaveOneOut()
y_proba = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]

    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ])
    inner_cv = GridSearchCV(pipe, param_grid, cv=5,
                            scoring='roc_auc', refit=True)
    inner_cv.fit(X_train, y_train)
    y_proba[test_idx] = inner_cv.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y, y_proba)
if auc < 0.5:
    y_proba = 1 - y_proba
    auc = 1 - auc

print(f"\nAUC = {auc:.3f}")

# ============================================================
# YOUDEN'S J — optimal threshold
# ============================================================
fpr, tpr, thresholds_roc = roc_curve(y, y_proba)
J = tpr + (1 - fpr) - 1
best_idx    = np.argmax(J)
best_thresh = thresholds_roc[best_idx]
best_sens   = tpr[best_idx]
best_spec   = 1 - fpr[best_idx]
best_J      = J[best_idx]

print(f"Youden's optimal threshold: {best_thresh:.3f}")
print(f"Sensitivity: {best_sens:.3f}")
print(f"Specificity: {best_spec:.3f}")
print(f"J statistic: {best_J:.3f}")

# ============================================================
# PREDICTIONS AT YOUDEN THRESHOLD
# ============================================================
y_pred = (y_proba >= best_thresh).astype(int)

print(f"\n--- Collagen + Clinical (AUC={auc:.3f}, threshold={best_thresh:.3f}) ---")
print(classification_report(y, y_pred,
                            target_names=['Short survival', 'Long survival'],
                            zero_division=0))

# ============================================================
# FIGURE 1 — CONFUSION MATRIX
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5))
cm_vals = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_vals,
                               display_labels=['Short', 'Long'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f"Collagen + Clinical\n"
             f"AUC={auc:.2f} | threshold={best_thresh:.2f}\n"
             f"Sensitivity={best_sens:.2f} | Specificity={best_spec:.2f}",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\confusion_youden_final.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2 — ROC CURVE + YOUDEN POINT
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, linewidth=2.5, color='#1B7FCC',
        label=f'Collagen + Clinical (AUC={auc:.2f})')
ax.scatter(fpr[best_idx], tpr[best_idx],
           color='red', s=200, zorder=5,
           edgecolors='black', linewidths=1.5,
           label=f"Youden point\nthreshold={best_thresh:.2f}, J={best_J:.2f}")
ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random (AUC=0.50)')
ax.set_xlabel("False Positive Rate ", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\roc_youden_final.png',
            dpi=200, bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 3 — YOUDEN'S J vs THRESHOLD
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_roc[:-1], J[:-1], linewidth=2,
        color='#1B7FCC', label="Youden's J")
ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2,
           label=f'Optimal threshold = {best_thresh:.3f}')
ax.axhline(best_J, color='gray', linestyle=':', linewidth=1,
           label=f'Max J = {best_J:.3f}')
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel("Youden's J = Sensitivity + Specificity - 1", fontsize=11)
ax.set_title("Youden's J vs Threshold\nCollagen + Clinical [Omni+Angle(mean+std) + AGE + Sex]",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\youden_curve_final.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved!")
print("  - confusion_youden_final.png")
print("  - roc_youden_final.png")
print("  - youden_curve_final.png")