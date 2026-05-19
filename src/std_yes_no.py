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


#  Load raw 675 images

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

# Mean and std per patient
pat_mean = df.groupby('pat')[['Omni Test', 'angle median']].mean()
pat_std  = df.groupby('pat')[['Omni Test', 'angle median']].std()
pat_mean.columns = ['Omni_mean', 'angle_median_mean']
pat_std.columns  = ['Omni_std',  'angle_median_std']
pat_df = pd.concat([pat_mean, pat_std], axis=1).reset_index()


# Load clinical

clinical = pd.read_csv(
    r'C:\Users\nechy\Downloads\BioImage_Project\data_clinical_patient.txt',
    sep='\t', comment='#', header=0)
clinical['OS_MONTHS']   = pd.to_numeric(clinical['OS_MONTHS'], errors='coerce')
clinical['AGE']         = pd.to_numeric(clinical['AGE'], errors='coerce')
clinical['T_STAGE_NUM'] = clinical['PATH_T_STAGE'].map({'T1':1,'T2':2,'T3':3,'T4':4})
clinical['N_STAGE_NUM'] = clinical['PATH_N_STAGE'].map({'N0':0,'N1':1,'N2':2})
clinical['SEX_NUM']     = clinical['SEX'].map({'Male':0,'Female':1})

merged = pat_df.merge(clinical[['PATIENT_ID','OS_MONTHS','AGE',
                                  'T_STAGE_NUM','N_STAGE_NUM','SEX_NUM']],
                      left_on='pat', right_on='PATIENT_ID')
median_os = merged['OS_MONTHS'].median()
merged['label'] = (merged['OS_MONTHS'] > median_os).astype(int)
y = merged['label'].values

print(f"Patients: {len(merged)}, Short=0: {(y==0).sum()}, Long=1: {(y==1).sum()}")


#  Feature sets: with and without std

imputer = SimpleImputer(strategy='median')

feature_sets = {
    'Imaging(mean) + AGE + T + N + Sex':
        imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                       'AGE', 'T_STAGE_NUM',
                                       'N_STAGE_NUM', 'SEX_NUM']].values),

    'Imaging(mean+std) + AGE + T + N + Sex':
        imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                       'Omni_std',  'angle_median_std',
                                       'AGE', 'T_STAGE_NUM',
                                       'N_STAGE_NUM', 'SEX_NUM']].values),

    'Imaging(mean) + AGE + T + N':
        imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                       'AGE', 'T_STAGE_NUM',
                                       'N_STAGE_NUM']].values),

    'Imaging(mean+std) + AGE + T + N':
        imputer.fit_transform(merged[['Omni_mean', 'angle_median_mean',
                                       'Omni_std',  'angle_median_std',
                                       'AGE', 'T_STAGE_NUM',
                                       'N_STAGE_NUM']].values),
}


# LOOCV, RobustScaler + SVM

param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C':      [0.01, 0.1, 0.5, 1.0]
}

loo = LeaveOneOut()
results = []

print(f"\n{'Feature set':<45} {'AUC':>7}")
print('-' * 55)

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

    auc  = roc_auc_score(y, y_proba)
    best = max(auc, 1 - auc)
    results.append((feat_name, y_proba, best))
    print(f"{feat_name:<45} {best:>7.3f}")


# ROC curve

results.sort(key=lambda x: x[2], reverse=True)
colors = ['#1B7FCC', '#D95F02', '#2CA02C', '#9467BD']

fig, ax = plt.subplots(figsize=(9, 7))

for i, (feat_name, y_proba, auc) in enumerate(results):
    if roc_auc_score(y, y_proba) < 0.5:
        y_proba = 1 - y_proba
    fpr, tpr, _ = roc_curve(y, y_proba)
    ax.plot(fpr, tpr, linewidth=2, color=colors[i],
            label=f'{feat_name}\n(AUC={auc:.2f})')

ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random (AUC=0.50)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('RobustScaler + SVM + Mean vs RobustScaler + SVM + Mean + Std',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\nechy\Downloads\BioImage_Project\results\roc_std_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved!")