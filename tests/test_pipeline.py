import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_data import generate_synthetic_data
from train_model import FEATURES, TARGET, train


def test_data_shape_and_columns():
    data = generate_synthetic_data(n_rows=200, seed=7)
    assert len(data) == 200
    assert set(FEATURES + [TARGET]).issubset(data.columns)
    assert not data.isna().any().any()


def test_model_trains_and_predicts_probabilities():
    data = generate_synthetic_data(n_rows=500, seed=11)
    model, metrics = train(data)
    probabilities = model.predict_proba(data[FEATURES].head(10))[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert 0.5 <= metrics["roc_auc"] <= 1.0

