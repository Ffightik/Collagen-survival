import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis
)

from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_predict
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD DATA
# ============================================================

wb = openpyxl.load_workbook(
    r"C:\Users\nechy\Downloads\BioImage_Project\eosine\CA_Out\Combined_statistics_ALLfibFeatures2.xlsx",
    read_only=True
)

ws = wb['CAcombined']
rows = list(ws.iter_rows(values_only=True))

df = pd.DataFrame(rows[1:], columns=rows[0])

wb.close()

# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# PATIENT ID
# ============================================================

df['pat'] = df['image label'].str.extract(
    r'eosine(TCGA-[A-Z0-9]+-[A-Z0-9]+)_\d+'
)

# ============================================================
# PATIENT FEATURES
# ============================================================

pat_mean = df.groupby('pat')[[
    'Omni Test',
    'angle median'
]].mean()

pat_std = df.groupby('pat')[[
    'Omni Test',
    'angle median'
]].std()

pat_mean.columns = [
    'Omni_mean',
    'angle_median_mean'
]

pat_std.columns = [
    'Omni_std',
    'angle_median_std'
]

pat_df = pd.concat(
    [pat_mean, pat_std],
    axis=1
).reset_index()

# ============================================================
# CLINICAL DATA
# ============================================================

clinical = pd.read_csv(
    r"C:\Users\nechy\Downloads\BioImage_Project\data_clinical_patient.txt",
    sep='\t',
    comment='#',
    header=0
)

clinical['OS_MONTHS'] = pd.to_numeric(
    clinical['OS_MONTHS'],
    errors='coerce'
)

clinical['AGE'] = pd.to_numeric(
    clinical['AGE'],
    errors='coerce'
)

clinical['SEX_NUM'] = clinical['SEX'].map({
    'Male': 0,
    'Female': 1
})

# ============================================================
# MERGE
# ============================================================

merged = pat_df.merge(
    clinical[['PATIENT_ID', 'OS_MONTHS', 'AGE', 'SEX_NUM']],
    left_on='pat',
    right_on='PATIENT_ID'
)

# ============================================================
# LABELS
# ============================================================

median_os = merged['OS_MONTHS'].median()

merged['label'] = (
    merged['OS_MONTHS'] > median_os
).astype(int)

y = merged['label'].values

# ============================================================
# FEATURES
# ============================================================

X = merged[[
    'Omni_mean',
    'angle_median_mean',
    'Omni_std',
    'angle_median_std',
    'AGE',
    'SEX_NUM'
]].values

print("\n===================================================")
print("DATASET")
print("===================================================")

print(f"Patients        : {len(y)}")
print(f"Short survival  : {(y==0).sum()}")
print(f"Long survival   : {(y==1).sum()}")

print("===================================================")

# ============================================================
# PIPELINE
# ============================================================

qda_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('clf', QuadraticDiscriminantAnalysis(
        reg_param=0.1
    ))
])

# ============================================================
# LOOCV
# ============================================================

loo = LeaveOneOut()

y_proba = cross_val_predict(
    estimator=qda_pipe,
    X=X,
    y=y,
    cv=loo,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(y, y_proba)

print("\n===================================================")
print("QUADRATIC DISCRIMINANT ANALYSIS")
print("===================================================")

print(f"AUC : {auc:.3f}")

# ============================================================
# THRESHOLD = 0.5
# ============================================================

y_pred = (y_proba >= 0.5).astype(int)

# ============================================================
# METRICS
# ============================================================

acc = accuracy_score(y, y_pred)

f1 = f1_score(y, y_pred)

prec = precision_score(y, y_pred)

rec = recall_score(y, y_pred)

cm = confusion_matrix(y, y_pred)

tn, fp, fn, tp = cm.ravel()

sens = tp / (tp + fn)

spec = tn / (tn + fp)

print("\n===================================================")
print("FINAL METRICS")
print("===================================================")

print(f"AUC         : {auc:.3f}")
print(f"Accuracy    : {acc:.3f}")
print(f"F1-score    : {f1:.3f}")
print(f"Precision   : {prec:.3f}")
print(f"Recall      : {rec:.3f}")
print(f"Sensitivity : {sens:.3f}")
print(f"Specificity : {spec:.3f}")

print("===================================================")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y,
        y_pred,
        target_names=[
            'Short survival',
            'Long survival'
        ],
        zero_division=0
    )
)

# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y, y_proba)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f'QDA ROC (AUC={auc:.2f})'
)

ax.plot(
    [0, 1],
    [0, 1],
    'k--',
    linewidth=1
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")

ax.set_title(
    "ROC Curve\nQuadratic Discriminant Analysis",
    fontsize=13,
    fontweight='bold'
)

ax.legend()

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(5, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Short', 'Long']
)

disp.plot(
    ax=ax,
    cmap='Blues',
    colorbar=False
)

ax.set_title(
    f"QDA\n"
    f"AUC={auc:.2f} | "
    f"Sens={sens:.2f} | "
    f"Spec={spec:.2f}",
    fontsize=11,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ============================================================
# BOOTSTRAP CI
# ============================================================

n_bootstraps = 2000

rng = np.random.RandomState(42)

bootstrapped_scores = []

for i in range(n_bootstraps):

    indices = rng.randint(0, len(y), len(y))

    if len(np.unique(y[indices])) < 2:
        continue

    score = roc_auc_score(
        y[indices],
        y_proba[indices]
    )

    bootstrapped_scores.append(score)

bootstrapped_scores = np.array(bootstrapped_scores)

ci_lower = np.percentile(
    bootstrapped_scores,
    2.5
)

ci_upper = np.percentile(
    bootstrapped_scores,
    97.5
)

print("\n===================================================")
print("BOOTSTRAP 95% CI")
print("===================================================")

print(f"AUC = {auc:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print("===================================================")

print("\nDone.")