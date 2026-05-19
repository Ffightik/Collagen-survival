"""
config.py — Central configuration file
Update paths below to match your local setup before running any script.
"""

import os

# ============================================================
# DATA PATHS — update these to your local paths
# ============================================================

# Root project directory
PROJECT_DIR = r'C:\Users\nechy\Downloads\BioImage_Project'

# CurveAlign output — fiber features per image
CURVEALIGN_XLSX = os.path.join(
    PROJECT_DIR,
    r'eosine\ctFIREout\CA_Out\Combined_statistics_ALLfibFeatures2.xlsx'
)

# Clinical data (TCGA-PAAD) — NOT included in repo
CLINICAL_TXT = os.path.join(PROJECT_DIR, 'data_clinical_patient.txt')

# CT-FIRE output folder
CTFIRE_DIR = os.path.join(PROJECT_DIR, r'eosine\ctFIREout')

# Binary masks output folder
BINARY_MASKS_DIR = os.path.join(PROJECT_DIR, r'eosine\BinaryMasks')

# Results output folder
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ============================================================
# MODEL PARAMETERS
# ============================================================

# Survival split
SURVIVAL_SPLIT_MONTHS = None   # None = use median OS automatically

# SVM parameters
SVM_C      = 0.1
SVM_KERNEL = 'rbf'

# Features to use
FEATURE_COLS_4 = ['Omni_mean', 'angle_median_mean', 'AGE', 'SEX_NUM']
FEATURE_COLS_6 = ['Omni_mean', 'angle_median_mean',
                  'Omni_std',  'angle_median_std',
                  'AGE', 'SEX_NUM']

# Random seed
RANDOM_STATE = 42

# ============================================================
# CREATE RESULTS DIR IF NOT EXISTS
# ============================================================
os.makedirs(RESULTS_DIR, exist_ok=True)