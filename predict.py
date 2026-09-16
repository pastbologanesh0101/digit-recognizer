"""
predict.py - Predict a digit from the held-out test set using the saved model.

Usage:
    python predict.py <index>

<index> is an index into the test split that was saved by train.py
(0 <= index < len(X_test)). Prints the predicted digit and the actual digit.
"""

import sys
import os
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <index>")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
    except ValueError:
        print("Error: <index> must be an integer.")
        sys.exit(1)

    try:
        predicted, actual = predict(index)
    except (FileNotFoundError, IndexError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Predicted digit: {predicted}")
    print(f"Actual digit:    {actual}")
    print("Match!" if predicted == actual else "Mismatch.")


if __name__ == "__main__":
    main()
