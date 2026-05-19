import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import openpyxl


# Load imaging features

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
feature_cols = [c for c in df.columns if c not in ['No.', 'image label', 'pat']]
pat_mean = df.groupby('pat')[feature_cols].mean().reset_index()


# Load and prepare clinical features

clinical = pd.read_csv(
    r'C:\Users\nechy\Downloads\BioImage_Project\data_clinical_patient.txt',
    sep='\t', comment='#', header=0)

clinical['OS_MONTHS']   = pd.to_numeric(clinical['OS_MONTHS'], errors='coerce')
clinical['AGE']         = pd.to_numeric(clinical['AGE'], errors='coerce')

clinical['T_STAGE_NUM'] = clinical['PATH_T_STAGE'].map(
    {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4})

clinical['N_STAGE_NUM'] = clinical['PATH_N_STAGE'].map(
    {'N0': 0, 'N1': 1, 'N2': 2})

clinical['AJCC_NUM'] = clinical['AJCC_PATHOLOGIC_TUMOR_STAGE'].map({
    'STAGE I': 1,  'STAGE IA': 1, 'STAGE IB': 1,
    'STAGE II': 2, 'STAGE IIA': 2, 'STAGE IIB': 2,
    'STAGE III': 3, 'STAGE IV': 4})

clinical['SEX_NUM']       = clinical['SEX'].map({'Male': 0, 'Female': 1})
clinical['RADIATION_NUM'] = clinical['RADIATION_THERAPY'].map({'No': 0, 'Yes': 1})

print("Clinical features sample:")
print(clinical[['PATIENT_ID', 'AGE', 'T_STAGE_NUM', 'N_STAGE_NUM',
                 'AJCC_NUM', 'SEX_NUM']].head(5).to_string())


#Merge imaging + clinical

merged = pat_mean.merge(clinical, left_on='pat', right_on='PATIENT_ID')
median_os = merged['OS_MONTHS'].median()
merged['label'] = (merged['OS_MONTHS'] > median_os).astype(int)
y = merged['label'].values

print(f"\nPatients: {len(merged)}, Median OS: {median_os:.1f} months")
print(f"Short=0: {(y==0).sum()}, Long=1: {(y==1).sum()}")


# Define feature sets

imputer = SimpleImputer(strategy='median')

feature_sets = {
    'Imaging only [Omni, angle_median]':
        imputer.fit_transform(merged[['Omni Test', 'angle median']].values),
    'Imaging + fiber':
    imputer.fit_transform(merged[['Omni Test', 'angle median', 'fiber weight']].values),

    'Imaging + AGE':
        imputer.fit_transform(merged[['Omni Test', 'angle median', 'AGE']].values),

    'Imaging + AGE + T_stage':
        imputer.fit_transform(merged[['Omni Test', 'angle median',
                                       'AGE', 'T_STAGE_NUM']].values),

    'Imaging + AGE + N_stage':
        imputer.fit_transform(merged[['Omni Test', 'angle median',
                                       'AGE', 'N_STAGE_NUM']].values),

    'Imaging + AGE + AJCC':
        imputer.fit_transform(merged[['Omni Test', 'angle median',
                                       'AGE', 'AJCC_NUM']].values),

    'Imaging + AGE + T + N':
        imputer.fit_transform(merged[['Omni Test', 'angle median',
                                       'AGE', 'T_STAGE_NUM', 'N_STAGE_NUM']].values),

    'Imaging + AGE + T + N + Sex':
        imputer.fit_transform(merged[['Omni Test', 'angle median', 'AGE',
                                       'T_STAGE_NUM', 'N_STAGE_NUM', 'SEX_NUM']].values),

'Imaging + AGE + Sex':
        imputer.fit_transform(merged[['Omni Test', 'angle median', 'AGE',
                                        'SEX_NUM']].values),
}

#LOOCV
param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C':      [0.01, 0.1, 0.5, 1.0]
}

loo = LeaveOneOut()
results = []

print(f"\n{'Feature set':<40} {'AUC':>7}")
print('-' * 50)

for feat_name, X in feature_sets.items():
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

    auc     = roc_auc_score(y, y_proba)
    auc_inv = roc_auc_score(y, 1 - y_proba)
    best    = max(auc, auc_inv)

    results.append((feat_name, y_proba, best))
    print(f"{feat_name:<40} {best:>7.3f}")

#ROC curve

results.sort(key=lambda x: x[2], reverse=True)

colors = ['#1B7FCC', '#D95F02', '#2CA02C', '#9467BD',
          '#8C564B', '#E377C2', '#7F7F7F', '#17BECF']

fig, ax = plt.subplots(figsize=(9, 7))

for i, (feat_name, y_proba, auc) in enumerate(results):
    auc_raw = roc_auc_score(y, y_proba)
    if auc_raw < 0.5:
        y_proba = 1 - y_proba
    fpr, tpr, _ = roc_curve(y, y_proba)
    ax.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
            label=f'{feat_name} (AUC={auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Nested LOOCV\nImaging vs Clinical vs Combined (TCGA-PAAD, n=33)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\roc_nested_final.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: roc_nested_final.png")