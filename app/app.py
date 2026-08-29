import os

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Failure Survival Risk",
    page_icon="🫀",
    layout="centered"
)

# ------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(APP_DIR, "heart_failure_model.pkl"))
    scaler = joblib.load(os.path.join(APP_DIR, "heart_failure_scaler.pkl"))
    feature_order = joblib.load(os.path.join(APP_DIR, "heart_failure_features.pkl"))
    return model, scaler, feature_order

model, scaler, feature_order = load_artifacts()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🫀 Heart Failure Survival Risk Estimator")
st.caption("A Logistic Regression model trained on the Heart Failure Clinical Records dataset (299 patients)")

st.warning(
    "**Research & educational tool only — not a clinical diagnostic device.** "
    "This model was trained on a small dataset (299 patients) and is intended to "
    "demonstrate a machine learning workflow, not to inform real medical decisions. "
    "Always consult a qualified healthcare professional for medical concerns.",
    icon="⚠️"
)

st.divider()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
st.subheader("Patient Clinical Measurements")
st.caption("Enter values for the four factors this model uses, based on formal statistical testing (ANOVA) and clinical relevance.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18, max_value=110, value=60, step=1,
        help="Patient's age at diagnosis."
    )
    ejection_fraction = st.slider(
        "Ejection Fraction (%)",
        min_value=10, max_value=80, value=38,
        help="Percentage of blood leaving the heart each contraction. "
             "Normal is roughly 55%+; below 30% is considered severely reduced."
    )

with col2:
    serum_creatinine = st.number_input(
        "Serum Creatinine (mg/dL)",
        min_value=0.3, max_value=10.0, value=1.1, step=0.1, format="%.1f",
        help="A marker of kidney function. Normal range is roughly 0.6–1.2 mg/dL; "
             "higher values indicate worsening kidney function, a known heart failure risk amplifier."
    )
    serum_sodium = st.number_input(
        "Serum Sodium (mEq/L)",
        min_value=110, max_value=150, value=137, step=1,
        help="Normal range is roughly 135–145 mEq/L. Low sodium (hyponatremia) "
             "is a known poor-prognosis marker in heart failure."
    )

st.divider()

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Estimate Survival Risk", type="primary", use_container_width=True):

    # Build input in the exact column order the scaler/model expect
    input_dict = {
        "serum_creatinine": serum_creatinine,
        "ejection_fraction": ejection_fraction,
        "age": age,
        "serum_sodium": serum_sodium,
    }
    input_df = pd.DataFrame([input_dict])[feature_order]

    # Scale using the SAME scaler fit during training — never re-fit here
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    prob_survived = probability[0]
    prob_died = probability[1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"**Higher risk profile flagged** — estimated {prob_died:.0%} probability of mortality during follow-up, "
                 f"based on patterns in this training dataset.")
    else:
        st.success(f"**Lower risk profile flagged** — estimated {prob_survived:.0%} probability of survival during follow-up, "
                   f"based on patterns in this training dataset.")

    # Probability bar
    st.write("Predicted probability breakdown:")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Survived", f"{prob_survived:.1%}")
    with prob_col2:
        st.metric("Died (during follow-up)", f"{prob_died:.1%}")

    st.progress(float(prob_died))

    st.caption(
        "On held-out test data, this model correctly identified 76% of patients who died "
        "during follow-up (recall), with 56% precision on that class — meaning it is tuned "
        "to catch at-risk patients, at the cost of some false alarms. It should not be "
        "interpreted as a precise individual probability."
    )

st.divider()

# ------------------------------------------------------------------
# About / methodology section
# ------------------------------------------------------------------
with st.expander("About this model"):
    st.markdown("""
    This app is built on the best-performing model from a broader comparison of four
    classifier families — Logistic Regression, Polynomial Logistic Regression, Support
    Vector Machines, and K-Nearest Neighbors — evaluated on the
    [Heart Failure Clinical Records dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
    (299 patients).

    **Why these four features?** An ANOVA F-test (computed on the training split only,
    to avoid data leakage) identified `serum_creatinine`, `ejection_fraction`, and
    `serum_sodium` as statistically significant predictors of mortality (p < 0.05).
    `age` was retained despite a borderline p-value (0.098) given its established
    clinical relevance and its 4th-highest F-score.

    **Why Logistic Regression?** Across every model tested, a simple linear decision
    boundary matched or outperformed more complex approaches (polynomial terms, kernel
    methods, distance-based voting) — evidence that the survival signal in this dataset
    is largely linear rather than requiring non-linear modeling.

    **Test set performance (4-feature Logistic Regression):**
    - Accuracy: 73%
    - Recall (died): 76%
    - Precision (died): 56%
    - ROC-AUC: 0.78

    **Limitations:** This is a small (299-patient), single-source, non-externally-validated
    dataset. Findings are hypothesis-generating, not clinically conclusive, and this tool
    is not a substitute for professional medical judgment.
    """)

st.caption("Built as part of a data science project comparing classification approaches on clinical survival data.")
