# Diabetes Risk Prediction with MLPs

**Author:** Adel Ahmed Fathi Amer 

Predicts a diabetes diagnosis (`diagnosed_diabetes`) with **multi-layer perceptrons only**, using the Kaggle dataset
[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)
(100,000 *synthetic* patient records).

| File | What it is |
|---|---|
| `diabetes_risk_mlp.ipynb` | The full study, all cells executed (data analysis and leakage check, feature sets, preprocessing, architecture, activation, regularisation, optimiser, loss, random search, final test evaluation, calibration, importance, subgroups) |
| `diabetes_risk_report.pdf` | 10-page report with the results and their interpretation |
| `results/all_runs.csv` | All 240 evaluated configurations |
| `results/final_results.json` | Final metrics, confusion matrices, importance, subgroup results |
| `figs/` | All figures produced by the notebook |

Two scenarios are studied: **S1** (with lab tests: glucose, insulin, HbA1c) and **S2** (without lab tests, a screening setting).
`diabetes_stage` is excluded from every model because it leaks the target (a model given it reaches 99.9 % accuracy).

## Results (held-out test set, 15,000 patients; ensemble of 5 MLPs)

| | S1 - with lab tests | S2 - without lab tests |
|---|---|---|
| Accuracy (95 % CI) | **92.0 %** (91.6 - 92.5) | **62.8 %** (62.0 - 63.5) |
| ROC-AUC (95 % CI) | **0.944** (0.941 - 0.948) | **0.652** (0.643 - 0.660) |
| Sensitivity / specificity | 0.869 / 0.998 | 0.810 / 0.354 |
| Always predict "diabetic" | 60.0 % | 60.0 % |
| Clinical rule HbA1c >= 6.5 | 91.0 % | - |

* With lab tests, the MLP essentially re-discovers the clinical HbA1c >= 6.5 rule (HbA1c is by far the most important input).
* Without lab tests the signal is weak (family history, age and physical activity); the model can rank patients for screening but not diagnose.
* The networks do not over-fit; changes of architecture, regularisation and optimiser move AUC by less than 0.002. The data, not the model, is the limit.
* The data are synthetic: do not use these models for medical decisions.

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook diabetes_risk_mlp.ipynb      # run all cells (~45 min on a GPU); the dataset is fetched with kagglehub
```

The dataset is **not** included; `kagglehub` downloads it (about 14 MB) on first run.
