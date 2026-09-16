"""
train.py - Train a handwritten digit classifier on sklearn's load_digits dataset.

Usage:
    python train.py

Trains an SVM classifier on the 8x8 pixel digit images bundled with
scikit-learn, evaluates it on a held-out test split, prints the resulting
accuracy, and saves both the trained model and the test split to disk
(model.joblib) so predict.py can reuse the exact same test set.
"""

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODEL_PATH = "model.joblib"
RANDOM_STATE = 42


def train_and_evaluate():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = SVC(kernel="rbf", gamma=0.001, C=10)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    joblib.dump(
        {
            "model": clf,
            "X_test": X_test,
            "y_test": y_test,
            "accuracy": accuracy,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model and test split to {MODEL_PATH}")

    return accuracy


if __name__ == "__main__":
    train_and_evaluate()
