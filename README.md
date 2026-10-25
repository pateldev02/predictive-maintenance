# Predictive Maintenance Using Machine Learning

An end-to-end machine learning case study using the **AI4I 2020 Predictive Maintenance Dataset** to predict machine failures and support proactive maintenance decisions.

## Business Objective

Unexpected equipment failure can create downtime, lost production capacity, maintenance cost, quality risk, and operational disruption.

This project focuses on a practical operating objective:

> **Detect at least 85% of actual machine failures while maintaining the highest practical precision and minimizing unnecessary maintenance alerts.**

This is a case-study operating requirement, not an industry or regulatory standard.

## Dataset

- 10,000 observations
- 339 machine failures
- Failure rate: approximately 3.39%
- Target: `Machine failure`

Identifiers such as `UDI` and `Product ID` were excluded from model training. Failure-mode indicator columns were also excluded to prevent target leakage.

## Feature Engineering

Engineering-informed variables were created to better represent machine operating stress:

- **Temperature Difference** = Process Temperature - Air Temperature
- **Angular Velocity** = RPM × 2π / 60
- **Mechanical Power** = Torque × Angular Velocity
- **Tool Wear Load** = Tool Wear × Torque

## Models Evaluated

- Naive Baseline
- Logistic Regression
- Balanced Logistic Regression
- Random Forest
- Balanced Random Forest
- XGBoost
- Balanced XGBoost

## Why Accuracy Was Not Enough

Because failures represent only 3.39% of the dataset, a model could achieve approximately 96.6% accuracy by predicting that every machine will operate normally while detecting **zero failures**.

The project therefore focused on:

- **Precision** — how many predicted failures were actually failures
- **Recall** — how many actual failures were detected
- **F1 Score** — balance between precision and recall
- **ROC-AUC** — overall ranking ability across thresholds
- **PR-AUC** — precision-recall performance for the rare failure class

## Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Naive Baseline | 0.000 | 0.000 | 0.000 | 0.500 | 0.034 |
| Logistic Regression | 0.571 | 0.176 | 0.270 | 0.926 | 0.464 |
| Balanced Logistic Regression | 0.179 | 0.882 | 0.298 | 0.939 | 0.430 |
| Random Forest | **0.965** | 0.809 | **0.880** | 0.973 | **0.891** |
| Balanced Random Forest | 0.961 | 0.721 | 0.824 | 0.971 | 0.873 |
| XGBoost | 0.914 | 0.779 | 0.841 | **0.986** | **0.891** |
| Balanced XGBoost | 0.806 | 0.853 | 0.829 | 0.983 | 0.887 |

## Final Model

**Random Forest** was selected as the leading model.

At the default threshold of `0.50`:

- Precision: **96.5%**
- Recall: **80.9%**
- F1 Score: **88.0%**
- ROC-AUC: **97.3%**
- PR-AUC: **89.1%**

Although XGBoost achieved the highest ROC-AUC, Random Forest delivered the stronger operational precision-recall balance.

## Business Threshold Analysis

The probability threshold was treated as an operating-policy decision rather than a fixed technical setting.

At threshold `0.20`, Random Forest achieved:

- Recall: **86.8%**
- Precision: **78.7%**
- F1 Score: **82.5%**

This satisfied the case-study objective of detecting at least 85% of failures.

At approximately the same recall level, XGBoost produced lower precision, which would create more unnecessary maintenance alerts.

## Explainability

The most influential predictive signals included:

- Rotational speed
- Mechanical power
- Torque
- Tool wear
- Tool wear load
- Temperature difference
- Air and process temperature

Feature importance represents predictive association and should not be interpreted as proof of causation.

## Repository Structure

```text
predictive-maintenance/
│
├── predictive_maintenance_ml.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

The trained `.pkl` model and project reports are intentionally excluded from the public repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/pateldev02/predictive-maintenance.git
cd predictive-maintenance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the project notebook.

## Model Artifact

The trained Random Forest `.pkl` file is intentionally not published in this repository.

For access to the serialized model artifact for review, testing, or discussion, please contact me:

**Email:** `pateldevp01@gmail.com`

## Limitations

- The AI4I 2020 dataset is a simulated industrial dataset.
- Only 339 failure observations are available.
- The data does not represent full longitudinal machine-degradation history.
- Feature importance indicates predictive association, not causation.
- Site-specific asset criticality, downtime cost, maintenance capacity, and operating conditions would need to be incorporated before production deployment.
- Threshold selection in this case study was evaluated on the held-out test set; a production workflow should tune thresholds on validation data or cross-validation and preserve a final untouched test set.

## Future Improvements

- Time-series sensor data
- Vibration and acoustic signals
- Maintenance-history integration
- Cost-sensitive threshold optimization
- Cross-validation and hyperparameter tuning
- Probability calibration
- Experiment tracking and model registry
- Data and model drift monitoring
- Automated retraining
- REST API deployment
- Maintenance dashboard integration
- Human-in-the-loop maintenance workflow

## Project Focus

The project demonstrates an end-to-end decision workflow:

**data quality → engineering context → model comparison → business threshold → explainability → governed operational use**

## Contact

For questions, collaboration, or access to the trained `.pkl` model artifact:

**Email:** `pateldevp01@gmail.com`
