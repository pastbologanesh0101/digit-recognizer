# AI Handwriting Recognition

A small, self-contained handwritten digit classifier built with scikit-learn.
It trains a Support Vector Machine (SVM) on the `load_digits` dataset that
ships with scikit-learn (1,797 samples of 8x8 grayscale images of digits
0-9), evaluates it on a held-out test split, and lets you run predictions on
individual test samples from the command line.

No external data download is required — `load_digits` is bundled with
scikit-learn.

## What it does

- `train.py` loads the digits dataset, splits it 80/20 into train/test sets,
  trains an `SVC` (RBF kernel) classifier, prints the resulting test
  accuracy, and saves the trained model plus the test split to
  `model.joblib`.
- `predict.py <index>` loads the saved model and test split, runs a
  prediction on the sample at `<index>`, and prints the predicted digit next
  to the actual (ground-truth) digit.

## Example output

```
$ python train.py
Test accuracy: 99.17%
Saved trained model and test split to model.joblib

$ python predict.py 0
Predicted digit: 5
Actual digit:    5
Match!

$ python predict.py 5
Predicted digit: 2
Actual digit:    2
Match!
```

Accuracy will vary slightly depending on the scikit-learn version and
platform, but consistently comes out above 95% with the default settings
(a fixed `random_state=42` is used for the train/test split).

## How to run

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (prints test accuracy, saves model.joblib)
python train.py

# 4. Predict on a held-out test sample by index
python predict.py 0
python predict.py 42
```

Valid indices for `predict.py` range from `0` to `len(X_test) - 1` (about
359 samples with the default 80/20 split of the 1,797-sample dataset).
Passing an out-of-range index prints a clear error instead of crashing.

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```

The test suite (`tests/test_digit_recognizer.py`) trains the model once and
checks:

1. `train.py` actually produces a saved `model.joblib` file.
2. Test accuracy exceeds a 90% threshold.
3. `predict()` returns a digit in the valid `0-9` range for a single sample.
4. `predict()` returns valid digits across multiple sample indices.
5. `predict()` raises a clear `IndexError` for an out-of-range index.
6. Predictions match the ground truth for the large majority of samples
   (consistent with the measured accuracy).

## Project structure

```
digit-recognizer/
├── train.py                        # Train + evaluate + save the model
├── predict.py                      # Load model, predict a single sample
├── tests/
│   └── test_digit_recognizer.py    # Unit tests (pytest)
├── requirements.txt
├── .github/workflows/tests.yml     # CI: runs pytest on push/PR
├── model.joblib                    # Saved trained model (see note below)
├── LICENSE
└── README.md
```

## Troubleshooting / FAQ

**`predict.py` says "Could not find 'model.joblib'"**
`model.joblib` is committed to the repo, so this normally only happens if
you deleted it, cloned a fork/branch where it's missing, or ran
`predict.py` from a different working directory than the repo root (it's
looked up as the relative path `model.joblib`). Run `python train.py` from
the repo root to regenerate it.

**Why does accuracy vary slightly between runs of `train.py`?**
The train/test split uses a fixed `random_state=42`, so the split itself is
deterministic. Small accuracy differences (typically well under 1%) can
still show up across different scikit-learn or BLAS/OpenMP library
versions, since `SVC`'s underlying `libsvm` solver isn't guaranteed to be
bit-for-bit identical across builds. Accuracy should always land above
95% with the default settings regardless.

**Why is `model.joblib` committed to git instead of `.gitignore`d?**
Most ML projects gitignore trained models because they're large. This one
doesn't, on purpose: `load_digits` images are only 8x8 pixels and the
resulting `SVC` model is small, so `model.joblib` is well under 1MB. See
"Note on the committed model file" below for details.

**I passed an out-of-range or negative index to `predict.py` — why the error?**
`predict.py <index>` only accepts indices into the held-out test split
saved by `train.py` (`0` to `len(X_test) - 1`, about 359 values by
default). Negative indices are rejected explicitly rather than falling
back to Python's "wrap to the end of the list" behavior, since that could
silently return a misleading prediction instead of a clear error.

## Note on the committed model file

`model.joblib` is committed directly to the repository rather than
`.gitignore`d. Because `load_digits` images are tiny (8x8 pixels) and the
SVM model itself is small, the resulting file is well under 1MB, so
committing it is harmless and lets anyone clone the repo and immediately
run `predict.py` without retraining first. Running `train.py` will
regenerate and overwrite it.

## Tech stack

- Python 3
- scikit-learn (`SVC`, `load_digits`, `train_test_split`)
- joblib (model persistence)
- pytest (unit tests)
- GitHub Actions (CI)

## License

MIT — see [LICENSE](LICENSE).
