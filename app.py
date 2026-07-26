"""Interactive Streamlit app for the synthetic AML risk explorer."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
DATA_PATH = ROOT / "data" / "synthetic_aml.csv"

LABELS = {
    "age": "Age",
    "wbc_10e9_l": "WBC (×10⁹/L)",
    "hemoglobin_g_dl": "Hemoglobin (g/dL)",
    "platelets_10e9_l": "Platelets (×10⁹/L)",
    "marrow_blast_percent": "Marrow blasts (%)",
    "complex_karyotype_1": "Complex karyotype",
    "monosomal_karyotype_1": "Monosomal karyotype",
    "tp53_abnormality_1": "TP53 abnormality",
    "flt3_itd_1": "FLT3-ITD",
    "npm1_mutation_1": "NPM1 mutation",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def feature_contributions(model, patient: pd.DataFrame) -> pd.DataFrame:
    transformed = model.named_steps["preprocessing"].transform(patient)
    names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    contributions = transformed[0] * coefficients
    clean_names = [name.split("__", 1)[-1] for name in names]
    result = pd.DataFrame(
        {
            "Feature": [LABELS.get(name, name.replace("_", " ").title()) for name in clean_names],
            "Contribution": contributions,
        }
    )
    result["Direction"] = result["Contribution"].map(
        lambda value: "Higher simulated risk" if value >= 0 else "Lower simulated risk"
    )
    return result.reindex(
        result["Contribution"].abs().sort_values(ascending=False).index
    ).head(8)


st.set_page_config(
    page_title="AML Cytogenetic Risk Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("Explainable AML Cytogenetic Risk Explorer")
st.caption(
    "Educational AI prototype using fully synthetic data — not for diagnosis, "
    "prognosis, treatment, or clinical decision-making."
)

if not (MODEL_PATH.exists() and METRICS_PATH.exists() and DATA_PATH.exists()):
    st.error("Project artifacts are missing. Run generate_data.py and train_model.py.")
    st.stop()

model = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
data = pd.read_csv(DATA_PATH)

with st.sidebar:
    st.header("Synthetic patient inputs")
    age = st.slider("Age", 18, 90, 55)
    wbc = st.number_input("WBC (×10⁹/L)", 0.4, 250.0, 25.0, 0.5)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", 4.5, 15.5, 9.5, 0.1)
    platelets = st.number_input("Platelets (×10⁹/L)", 5, 500, 70)
    blasts = st.slider("Marrow blasts (%)", 20.0, 99.0, 45.0, 0.5)
    complex_karyotype = st.checkbox("Complex karyotype")
    monosomal_karyotype = st.checkbox("Monosomal karyotype")
    tp53 = st.checkbox("TP53 abnormality")
    flt3 = st.checkbox("FLT3-ITD")
    npm1 = st.checkbox("NPM1 mutation")

patient = pd.DataFrame(
    [
        {
            "age": age,
            "wbc_10e9_l": wbc,
            "hemoglobin_g_dl": hemoglobin,
            "platelets_10e9_l": platelets,
            "marrow_blast_percent": blasts,
            "complex_karyotype": int(complex_karyotype),
            "monosomal_karyotype": int(monosomal_karyotype),
            "tp53_abnormality": int(tp53),
            "flt3_itd": int(flt3),
            "npm1_mutation": int(npm1),
        }
    ]
)

probability = float(model.predict_proba(patient)[0, 1])
if probability < 0.33:
    band = "Lower"
elif probability < 0.66:
    band = "Intermediate"
else:
    band = "Higher"

left, right = st.columns([1, 2])
with left:
    st.subheader("Model output")
    st.metric("Simulated adverse-risk probability", f"{probability:.1%}")
    st.metric("Educational probability band", band)
    st.info(
        "This output describes a synthetic model simulation. It does not reproduce "
        "a clinical guideline or validated risk score."
    )

with right:
    st.subheader("Why did the model produce this result?")
    contributions = feature_contributions(model, patient)
    fig = px.bar(
        contributions.sort_values("Contribution"),
        x="Contribution",
        y="Feature",
        color="Direction",
        orientation="h",
        color_discrete_map={
            "Higher simulated risk": "#d95f59",
            "Lower simulated risk": "#2f7f77",
        },
    )
    fig.update_layout(
        legend_title_text="Influence",
        xaxis_title="Contribution to model score",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Held-out test performance")
metric_columns = st.columns(5)
for column, (label, key) in zip(
    metric_columns,
    [
        ("ROC AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
    ],
):
    column.metric(label, f"{metrics[key]:.3f}")

st.subheader("Synthetic dataset overview")
chart_data = (
    data.groupby(["complex_karyotype", "tp53_abnormality"], as_index=False)[
        "adverse_risk"
    ]
    .mean()
    .rename(columns={"adverse_risk": "adverse_risk_rate"})
)
chart_data["Group"] = chart_data.apply(
    lambda row: (
        f"Complex: {'Yes' if row['complex_karyotype'] else 'No'} | "
        f"TP53: {'Yes' if row['tp53_abnormality'] else 'No'}"
    ),
    axis=1,
)
overview = px.bar(
    chart_data,
    x="Group",
    y="adverse_risk_rate",
    labels={"adverse_risk_rate": "Synthetic adverse-risk rate", "Group": ""},
    color="adverse_risk_rate",
    color_continuous_scale="Blues",
)
overview.update_yaxes(tickformat=".0%")
st.plotly_chart(overview, use_container_width=True)

with st.expander("Responsible-AI limitations"):
    st.markdown(
        """
- The dataset is generated by code and contains no real patients.
- Relationships are simplified and do not represent clinical truth.
- There is no external or prospective validation.
- The model is not calibrated or approved for healthcare use.
- No diagnosis, prognosis, or treatment decision should be based on this app.
"""
    )

