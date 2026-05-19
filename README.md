# Collagen-Survival

## Collagen Fiber Organization Analysis in Pancreatic Cancer Stroma Using Curvelet Transform and CT-FIRE

This project investigates collagen fiber organization in stromal regions surrounding pancreatic cancer using histopathology H&E images from the TCGA cohort. The primary goal was to determine whether collagen architecture differs between patients with short survival and long survival, and whether these differences can be used for survival prediction with machine learning models.

The analysis was performed on stromal ROIs extracted from H&E slides of 33 pancreatic cancer patients. Since collagen fibers are highly represented in the eosin component of H&E staining, the workflow focused on eosin-rich stromal structures and their geometric organization.

The project combines classical image analysis, collagen fiber extraction, handcrafted feature engineering, statistical analysis, and machine learning classification.

---

# Methodology

The main analysis pipeline was based on Curvelet Transform and CT-FIRE.

Curvelet Transform was selected because it is highly effective for representing elongated curvilinear structures such as collagen fibers. Compared to traditional texture operators, curvelets preserve directional and anisotropic information significantly better, making them especially useful for stromal collagen analysis.

Features extracted from transformed images included:

- Orientation-related statistics
- Angular distribution measurements
- Alignment descriptors
- Omni Test metrics
- Fiber organization measurements

CT-FIRE was used for collagen fiber reconstruction and quantitative fiber analysis.

---

# CT-FIRE Technical Issues and Solutions

Another major part of the work involved reconstructing binary collagen masks from CT-FIRE `.mat` outputs.

Several important problems were identified and corrected:

- Incorrect coordinate arrays (`Xa/Fa`) were initially used instead of real fiber coordinates (`Xai/Fai`)
- MATLAB indexing caused X/Y axis inversion
- Fiber index structures `Fa` and `Fai` were accidentally mixed

Correcting these issues allowed proper reconstruction of collagen fiber masks corresponding to CT-FIRE overlays.

---

# Feature Comparison

Feature distributions between short-survival and long-survival groups were compared using box plots.

The strongest differences were observed in collagen orientation and angular organization features. Metrics such as:

- Omni Test
- Angle Median 

showed the best separation between the two survival groups.

---

# Machine Learning

Several machine learning models were evaluated for binary survival prediction:

- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- XGBoost

Due to the small dataset size, Leave-One-Subject-Out Cross Validation (LOOCV) was used to reduce overfitting and improve robustness.

Multiple combinations of imaging and clinical features were tested, including:

- AGE
- SEX
- Omni Test
- Angle Median

The best-performing model was SVM with RobustScaler normalization.

---

# Best Results

## SVM + RobustScaler

### Features Used
- Omni Test
- Angle Median
- AGE
- SEX

| Metric | Value |
|---|---|
| AUC | 0.996 |
| Accuracy | 0.818 |
| F1-score | 0.814 |
| Precision | 0.864 |
| Recall | 0.824 |
| Sensitivity | 1.000 |
| Specificity | 0.647 |

---

# Visualizations

The repository includes:

- ROC curves
- Confusion matrices
- Feature box plots
- Multi-model comparison plots
- Binary collagen masks

---

# Technologies

- Python
- MATLAB
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- CT-FIRE
- Curvelet Transform

---

# Dataset

- TCGA pancreatic cancer cohort
- 33 patients
- H&E stromal ROIs
- Binary survival classification:
  - Long survival
  - Short survival

---

# Future Work

Potential future improvements include:

- Deep learning feature extraction
- Multi-scale collagen analysis
- External validation cohorts
- Integration with clinical and molecular data

---

# Authors

Nataliia Nechyporenko [LinkedId:https://www.linkedin.com/in/nataliia-nechyporenko-b05351362/]

Prof. Arkadiusz Gertych [LinkedId:https://www.linkedin.com/in/arkadiusz-gertych-5801431b5/]

Bioimage analysis and AI research project focused on computational pathology and cancer prognosis.
