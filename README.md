# Explainable AML Cytogenetic Risk Explorer

An educational AI-in-healthcare portfolio project that demonstrates how
cytogenetic and laboratory features can be used in an interpretable machine
learning workflow.

The project connects molecular genetics, laboratory diagnostics, data analysis,
and responsible health AI. It uses **fully synthetic data** and is **not a
clinical decision-support system**.

## What this project demonstrates

- Reproducible synthetic health-data generation
- Exploratory analysis of AML-related features
- Logistic-regression modelling with preprocessing
- Patient-level explanations showing which features influenced a prediction
- Evaluation using ROC AUC, sensitivity, specificity, precision, and a
  confusion matrix
- Responsible-AI documentation and clear limitations
- A Streamlit interface for interactive exploration

## Project structure

```text
aml-risk-explorer/
├── app.py                 # Interactive Streamlit application
├── generate_data.py       # Reproducible synthetic dataset generator
├── train_model.py         # Training and evaluation pipeline
├── requirements.txt
├── data/
│   └── synthetic_aml.csv
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
└── tests/
    └── test_pipeline.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

Then open the local address shown by Streamlit, usually
`http://localhost:8501`.

## Run tests

```bash
pytest -q
```

## Model

The binary outcome, `adverse_risk`, is generated from a transparent simulation.
The model receives age, blood counts, blast percentage, and selected
cytogenetic/molecular indicators. Numeric variables are standardized and
categorical variables are one-hot encoded before logistic regression.

The app displays:

- predicted probability in the **synthetic simulation**;
- an educational low/intermediate/high band;
- the strongest positive and negative feature contributions.

These bands are demonstrations of model behaviour—not ELN classifications or
clinical recommendations.

## Responsible use and limitations

- All records are synthetic; no real patient information is included.
- The simulated relationships are simplified and may not reflect real-world
  prevalence, biology, treatment, or outcomes.
- The model has no external validation, prospective validation, calibration
  study, regulatory review, or clinical approval.
- Demographic performance and clinical utility cannot be inferred from this
  dataset.
- Predictions must not be used for diagnosis, prognosis, treatment, or patient
  care.

## Possible next steps

1. Add calibration curves and decision-curve analysis.
2. Compare logistic regression with a tree-based model.
3. Add subgroup performance analysis.
4. Replace synthetic data only with a properly licensed, de-identified public
   dataset and document its provenance.
5. Create a model card and a short project report.

## Suggested GitHub description

> Responsible and explainable AI prototype for exploring AML cytogenetic risk
> patterns using synthetic data, logistic regression, and Streamlit.

## Author

Saman Raftari — Molecular genetics, clinical diagnostics, and AI in healthcare.

