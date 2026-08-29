# Heart Failure Survival Prediction

A model comparison project on the Heart Failure Clinical Records dataset (299 patients), comparing how differently-shaped classifiers — linear, polynomial, kernel-based, and distance-based — predict patient mortality from clinical measurements taken at diagnosis.

**Headline finding:** a simple Logistic Regression model using four clinically-grounded features (serum creatinine, ejection fraction, age, serum sodium) outperformed every more complex model tested, correctly identifying 76% of patients who died during follow-up on a held-out test set.

## What's in this repo

| File                                | Description                                                                                                                                                                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Notebook.ipynb`                  | Full analysis: data cleaning, EDA, feature relevance (VIF + ANOVA), Kaplan-Meier survival curves, and four tuned classifiers (Logistic Regression, Polynomial Logistic Regression, KNN, SVC) evaluated on a held-out test set |
| `Heart_Failure_Final_Report.docx` | Written report summarizing findings, methodology, and honest limitations                                                                                                                                                      |
| `streamlit_app/`                  | Interactive app for exploring the final model's predictions                                                                                                                                                                   |
| `Heart_Failure_Project_Brief.pdf` | Original project scope and objectives                                                                                                                                                                                         |

## Key findings

- **Linear beats complex.** Every model that tried to capture non-linear structure (polynomial terms, SVM kernels, KNN's local voting) matched or underperformed a plain linear decision boundary — SVM's own hyperparameter search consistently picked a linear kernel over RBF or polynomial.
- **Feature selection helps, consistently.** A 4-feature model (chosen via ANOVA F-test, run on the training split only to avoid data leakage) matched or beat the full 11-feature version in every model family tested — most dramatically for KNN, whose recall collapsed from 0.52 to 0.14 with all 11 features included (the curse of dimensionality in action).
- **Three independent methods agree.** ANOVA testing, domain-informed tiered mortality-rate analysis, and Kaplan-Meier survival curves all converged on the same two dominant risk markers: **serum creatinine** (kidney function) and **ejection fraction** (heart pumping efficiency).

See the [full report](https://github.com/Prosper438/heart-failure-survival-prediction/blob/main/Report/Heart_Failure_Final_Report.docx) for detailed methodology, the complete model comparison table, and an honest account of the analytical issues caught and corrected along the way (a feature-selection leakage bug and a metric-gaming SVM collapse, among others).

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook Notebook.ipynb
```

Requires `heart_failure_clinical_records_dataset.csv` in the same directory (not included here — available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)).

## Running the app

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Model performance (test set)

| Model                                      | Accuracy | Precision (died) | Recall (died)  | ROC-AUC |
| ------------------------------------------ | -------- | ---------------- | -------------- | ------- |
| **Logistic Regression (4 features)** | 0.73     | 0.56             | **0.76** | 0.78    |
| Polynomial LogReg (4 features)             | 0.71     | 0.54             | 0.72           | 0.78    |
| Polynomial LogReg (11 features)            | 0.74     | 0.59             | 0.69           | 0.74    |
| Logistic Regression (11 features)          | 0.73     | 0.58             | 0.66           | 0.74    |
| SVC (4 features)                           | 0.72     | 0.56             | 0.62           | 0.76    |
| SVC (11 features)                          | 0.71     | 0.55             | 0.62           | 0.73    |
| KNN (4 features)                           | 0.73     | 0.60             | 0.52           | 0.69    |
| KNN (11 features)                          | 0.64     | 0.36             | 0.14           | 0.65    |

Recall on the "died" class was prioritized as the primary evaluation metric, given the clinical cost of missing an at-risk patient.

## Limitations

This project does not produce a clinically deployable diagnostic tool. The dataset is small (299 patients, single institution, no external validation), and findings should be read as hypothesis-generating rather than clinically conclusive. See the full report for a complete discussion of limitations.

## Dataset

[Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) — UCI Machine Learning Repository.
