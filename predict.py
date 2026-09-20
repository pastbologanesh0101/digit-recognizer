"""
predict.py - Predict a digit from the held-out test set using the saved model.

Usage:
    python predict.py <index> [--model-path PATH]

<index> is an index into the test split that was saved by train.py
(0 <= index < len(X_test)). Prints the predicted digit and the actual digit.

--model-path/-m lets you point at a model file other than the default
model.joblib -- e.g. to compare predictions across a couple of retrained
models saved under different names.
"""

import argparse
import os
import sys

import joblib

MODEL_PATH = "model.joblib"


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run 'python train.py' first to train "
            "and save the model."
        )
    return joblib.load(path)


def predict(index, data=None):
    """Predict the digit at `index` in the saved test split.

    Returns (predicted_digit, actual_digit).
    """
    if data is None:
        data = load_model()

    required_keys = {"model", "X_test", "y_test"}
    if not isinstance(data, dict) or not required_keys.issubset(data):
        raise ValueError(
            "model.joblib is missing expected data (model/X_test/y_test). "
            "It may be corrupted, truncated, or saved by an incompatible "
            "version of this project -- try re-running 'python train.py' "
            "to regenerate it."
        )

    model = data["model"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    if index < 0 or index >= len(X_test):
        raise IndexError(
            f"Index {index} out of range. Valid range: 0 to {len(X_test) - 1}."
        )

    sample = X_test[index].reshape(1, -1)
    predicted = int(model.predict(sample)[0])
    actual = int(y_test[index])

    return predicted, actual


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict a digit from the held-out test set."
    )
    parser.add_argument(
        "index",
        type=int,
        help="Index into the saved test split (0 <= index < len(X_test)).",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        default=MODEL_PATH,
        help=f"Path to the saved model file (default: {MODEL_PATH}).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    try:
        predicted, actual = predict(args.index, data=load_model(args.model_path))
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Predicted digit: {predicted}")
    print(f"Actual digit:    {actual}")
    print("Match!" if predicted == actual else "Mismatch.")


if __name__ == "__main__":
    main()
