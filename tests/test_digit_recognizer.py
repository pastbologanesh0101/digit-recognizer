"""
Unit tests for the AI Handwriting Recognition project.

Run with:
    pytest
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_and_evaluate, MODEL_PATH
from predict import predict, load_model


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    """Train once for the whole test module and clean up afterwards."""
    accuracy = train_and_evaluate()
    yield accuracy
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)


def test_model_file_created():
    assert os.path.exists(MODEL_PATH), "train.py should save a model.joblib file"


def test_accuracy_above_threshold(trained_model):
    accuracy = trained_model
    assert accuracy > 0.90, f"Expected accuracy > 90%, got {accuracy * 100:.2f}%"


def test_predict_returns_valid_digit():
    predicted, actual = predict(0)
    assert 0 <= predicted <= 9, "Predicted digit must be between 0 and 9"
    assert 0 <= actual <= 9, "Actual digit must be between 0 and 9"


def test_predict_multiple_indices_return_valid_digits():
    data = load_model()
    num_test_samples = len(data["X_test"])
    for i in range(0, min(10, num_test_samples)):
        predicted, actual = predict(i, data=data)
        assert isinstance(predicted, int)
        assert 0 <= predicted <= 9
        assert 0 <= actual <= 9


def test_predict_out_of_range_index_raises():
    data = load_model()
    num_test_samples = len(data["X_test"])
    with pytest.raises(IndexError):
        predict(num_test_samples + 1000, data=data)


def test_predict_matches_actual_most_of_the_time():
    """Given >90% accuracy, most predictions on the first N samples should match."""
    data = load_model()
    num_test_samples = len(data["X_test"])
    sample_size = min(50, num_test_samples)
    matches = 0
    for i in range(sample_size):
        predicted, actual = predict(i, data=data)
        if predicted == actual:
            matches += 1
    assert matches / sample_size > 0.8


def test_predict_negative_index_raises():
    """Negative indices must be rejected explicitly.

    Plain Python list/array indexing treats -1 as "last element", which
    would silently return a real (misleading) prediction instead of an
    error. predict() explicitly disallows negative indices, so this should
    raise IndexError rather than quietly returning a result.
    """
    data = load_model()
    with pytest.raises(IndexError):
        predict(-1, data=data)


def test_load_model_missing_file_raises_clear_error():
    """load_model() on a non-existent path should fail with a clear,
    actionable FileNotFoundError rather than a raw OS-level traceback."""
    missing_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "no_such_model.joblib"
    )
    assert not os.path.exists(missing_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        load_model(missing_path)

    # The message should point the user at the fix, not just say "not found".
    assert "train.py" in str(exc_info.value)
