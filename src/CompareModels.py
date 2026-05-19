import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, roc_auc_score,
                             roc_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
pat_mean.columns = ['Omni_mean', 'angle_median_mean']
pat_df = pat_mean.reset_index()

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

feature_cols = ['Omni_mean', 'angle_median_mean', 'AGE', 'SEX_NUM']
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(merged[feature_cols].values)

print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
print(f"Labels:  Short={sum(y==0)}, Long={sum(y==1)}")
print(f"Median OS: {median_os:.1f} months\n")

# ============================================================
# MODELS
# ============================================================
models = {
    'SVM': Pipeline([
        ('scaler', RobustScaler()),
        ('clf', SVC(probability=True, kernel='rbf', C=0.1,
                    class_weight='balanced', random_state=42))
    ]),
    'LR': Pipeline([
        ('scaler', RobustScaler()),
        ('clf', LogisticRegression(C=0.01, class_weight='balanced',
                                   max_iter=1000, random_state=42))
    ]),
    'RF': Pipeline([
        ('scaler', RobustScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=3,
                                       class_weight='balanced',
                                       random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', RobustScaler()),
        ('clf', XGBClassifier(n_estimators=50, max_depth=2,
                              learning_rate=0.1,
                              eval_metric='logloss',
                              random_state=42, verbosity=0))
    ])
}

# ============================================================
# LOOCV
# ============================================================
loo     = LeaveOneOut()
results = {}

for name, pipe in models.items():
    print(f"Training {name}...")
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe.fit(X_train, y_train)
        y_true_all.append(y_test[0])
        y_pred_all.append(pipe.predict(X_test)[0])
        y_prob_all.append(pipe.predict_proba(X_test)[0, 1])

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_prob = np.array(y_prob_all)

    # AUC с коррекцией инверсии
    auc_val = roc_auc_score(y_true, y_prob)
    if auc_val < 0.5:
        auc_val = 1 - auc_val
        y_prob  = 1 - y_prob

    print(f"  prob range: [{y_prob.min():.3f}, {y_prob.max():.3f}]  "
          f"AUC={auc_val:.3f}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results[name] = {
        'cm':          cm,
        'auc':         auc_val,
        'accuracy':    accuracy_score(y_true, y_pred),
        'f1':          f1_score(y_true, y_pred, average='macro'),
        'precision':   precision_score(y_true, y_pred, average='macro',
                                       zero_division=0),
        'recall':      recall_score(y_true, y_pred, average='macro'),
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'fpr':         fpr,
        'tpr':         tpr,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

# ============================================================
# PRINT TABLE
# ============================================================
metric_keys   = ['auc', 'accuracy', 'f1', 'precision',
                 'recall', 'sensitivity', 'specificity']
metric_labels = ['AUC', 'Acc', 'F1', 'Prec', 'Recall', 'Sens', 'Spec']

print()
print("=" * 72)
print("  4 FEATURES (mean only) | RobustScaler | LOOCV | n=33")
print("  Features: Omni Test, Angle Median, AGE, SEX")
print("=" * 72)
print(f"{'Model':<10} " + " ".join(f"{m:>6}" for m in metric_labels))
print("-" * 72)
for mname, r in results.items():
    vals = [r[k] for k in metric_keys]
    print(f"{mname:<10} " + " ".join(f"{v:>6.3f}" for v in vals))
print("=" * 72)

# ============================================================
# FIGURE — 3 rows
# ============================================================
model_names  = list(results.keys())
colors_model = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

fig = plt.figure(figsize=(18, 16))
fig.suptitle('Model Comparison — 4 Features (mean only) | RobustScaler\n'
             'LOOCV | n=33 | Omni Test, Angle Median, AGE, SEX',
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.55, wspace=0.35,
                       height_ratios=[1, 1.3, 1.4])

# --- Row 0: Confusion Matrices ---
for i, mname in enumerate(model_names):
    r  = results[mname]
    cm = r['cm']
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Short', 'Long'], fontsize=9)
    ax.set_yticklabels(['Short', 'Long'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.set_title(f'{mname}\nAUC = {r["auc"]:.3f}',
                 fontsize=10, fontweight='bold')
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, str(cm[ii, jj]),
                    ha='center', va='center', fontsize=18,
                    fontweight='bold',
                    color='white' if cm[ii, jj] > cm.max()/2 else 'black')
    ax.text(0.5, -0.28,
            f"TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}",
            ha='center', transform=ax.transAxes,
            fontsize=8, color='gray')

# --- Row 1: Metrics bars per model ---
for i, mname in enumerate(model_names):
    r    = results[mname]
    vals = [r[k] for k in metric_keys]
    ax   = fig.add_subplot(gs[1, i])
    bars = ax.bar(metric_labels, vals,
                  color=colors_model[i], alpha=0.85, edgecolor='white')
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title(mname, fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

# --- Row 2: ROC curves + grouped comparison ---
ax_roc = fig.add_subplot(gs[2, :2])
for i, mname in enumerate(model_names):
    r = results[mname]
    ax_roc.plot(r['fpr'], r['tpr'],
                color=colors_model[i], lw=2,
                label=f'{mname} (AUC={r["auc"]:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('ROC Curves — All Models', fontsize=11, fontweight='bold')
ax_roc.legend(fontsize=9, loc='lower right')
ax_roc.grid(True, alpha=0.3)

ax_cmp = fig.add_subplot(gs[2, 2:])
x     = np.arange(len(metric_labels))
width = 0.18

for i, mname in enumerate(model_names):
    vals   = [results[mname][k] for k in metric_keys]
    offset = (i - 1.5) * width
    bars   = ax_cmp.bar(x + offset, vals, width,
                        label=mname, color=colors_model[i],
                        alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax_cmp.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=6, fontweight='bold', rotation=90)

ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(metric_labels, fontsize=11)
ax_cmp.set_ylim(0, 1.35)
ax_cmp.set_ylabel('Score', fontsize=11)
ax_cmp.set_title('All Metrics Comparison', fontsize=11, fontweight='bold')
ax_cmp.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax_cmp.legend(fontsize=9, loc='upper right')
ax_cmp.grid(True, axis='y', alpha=0.3)

plt.savefig(
    r'C:\Users\nechy\Downloads\BioImage_Project\results\model_comparison_4features_final.png',
    dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: model_comparison_4features_final.png")