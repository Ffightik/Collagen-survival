import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.model_selection import (
    LeaveOneOut,
    GridSearchCV,
    StratifiedKFold,
    permutation_test_score
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
# EXTRACT PATIENT ID
# ============================================================

df['pat'] = df['image label'].str.extract(
    r'eosine(TCGA-[A-Z0-9]+-[A-Z0-9]+)_\d+'
)

# ============================================================
# PATIENT-LEVEL FEATURES
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
# LOAD CLINICAL DATA
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
print(f"Patients: {len(y)}")
print(f"Short survival: {(y==0).sum()}")
print(f"Long survival : {(y==1).sum()}")
print("===================================================")

# ============================================================
# PIPELINE
# ============================================================

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('clf', SVC(probability=True, random_state=42))
])

# ============================================================
# PARAM GRID
# ============================================================

param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [0.01, 0.1, 0.5, 1.0],
    'clf__class_weight': ['balanced', None]
}

# ============================================================
# NESTED LOOCV
# ============================================================

loo = LeaveOneOut()

y_proba = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        refit=True,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_proba[test_idx] = grid.predict_proba(X_test)[:, 1]

# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(y, y_proba)

print(f"\nNested LOOCV ROC-AUC = {auc:.3f}")
# ============================================================
# BOOTSTRAP 95% CONFIDENCE INTERVAL FOR AUC
# ============================================================

from sklearn.metrics import roc_auc_score

n_bootstraps = 2000
rng = np.random.RandomState(42)

bootstrapped_scores = []

for i in range(n_bootstraps):

    # random sampling with replacement
    indices = rng.randint(0, len(y), len(y))

    # need both classes present
    if len(np.unique(y[indices])) < 2:
        continue

    score = roc_auc_score(
        y[indices],
        y_proba[indices]
    )

    bootstrapped_scores.append(score)

bootstrapped_scores = np.array(bootstrapped_scores)

# 95% CI
ci_lower = np.percentile(bootstrapped_scores, 2.5)
ci_upper = np.percentile(bootstrapped_scores, 97.5)

print("\n===================================================")
print("BOOTSTRAP 95% CONFIDENCE INTERVAL")
print("===================================================")

print(f"AUC = {auc:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print("===================================================")

# ============================================================
# PRIMARY THRESHOLD = 0.5
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

# ============================================================
# PRINT METRICS
# ============================================================

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
    label=f'SVM ROC (AUC={auc:.2f})'
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
    "ROC Curve\nNested LOOCV",
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
    f"SVM Nested LOOCV\n"
    f"AUC={auc:.2f} | "
    f"Sens={sens:.2f} | "
    f"Spec={spec:.2f}",
    fontsize=11,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ============================================================
# PERMUTATION TEST
# ============================================================

print("\nRunning permutation test...")

score, perm_scores, pvalue = permutation_test_score(
    estimator=pipe,
    X=X,
    y=y,
    scoring='roc_auc',
    cv=loo,
    n_permutations=1000,
    n_jobs=-1,
    random_state=42
)

print("\n===================================================")
print("PERMUTATION TEST")
print("===================================================")

print(f"Observed AUC : {score:.3f}")
print(f"P-value      : {pvalue:.5f}")

print("===================================================")

# ============================================================
# PERMUTATION HISTOGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    perm_scores,
    bins=30,
    alpha=0.8
)

ax.axvline(
    score,
    linestyle='--',
    linewidth=2,
    label=f'Observed AUC = {score:.3f}'
)

ax.set_xlabel("Permutation ROC-AUC")
ax.set_ylabel("Count")

ax.set_title(
    "Permutation Test Distribution",
    fontsize=12,
    fontweight='bold'
)

ax.legend()

plt.tight_layout()
plt.show()

print("\nDone.")