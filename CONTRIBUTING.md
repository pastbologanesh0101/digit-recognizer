# Contributing to digit-recognizer

Thanks for taking a look at this project. It's small and self-contained, so
the workflow is simple.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Retraining the model

```bash
python train.py
```

This loads scikit-learn's bundled `load_digits` dataset, trains a fresh
`SVC` classifier on an 80/20 train/test split, prints the resulting test
accuracy, and overwrites `model.joblib` with the new model plus the exact
test split used to evaluate it. Do this whenever you change anything in
`train.py` (e.g. the model, the split, or the hyperparameters).

## Running predictions

```bash
python predict.py <index>
```

`<index>` must be a valid index into the test split saved by `train.py`
(run `train.py` at least once first). See `README.md` for the valid range
and example output.

## Running the tests

```bash
pytest -v
```

Please run the full suite before opening a pull request. The suite trains
the model once (via a shared fixture) and exercises both `train.py` and
`predict.py`, including error paths (missing model file, out-of-range
index, etc.) — not just the happy path.

## Code style

- Keep `train.py` and `predict.py` dependency-light and readable; this
  project intentionally avoids frameworks and extra abstraction layers.
- Match the existing style: plain functions (not classes) for `train.py`
  and `predict.py`, docstrings on every public function, and `snake_case`
  naming throughout.
- Keep CLI error handling explicit — catch specific exceptions
  (`FileNotFoundError`, `IndexError`, `ValueError`) in `main()` and print a
  short, actionable message instead of letting a traceback leak to the
  user.
- Favor small, focused functions that are easy to unit test in isolation
  (e.g. `predict()` accepts an optional pre-loaded `data` dict specifically
  so tests don't have to reload the model file for every case).

## Submitting changes

1. Fork the repo and create a branch for your change.
2. Make your change, retrain if you touched `train.py`, and run `pytest -v`
   until it's green.
3. Update `README.md` (and `CHANGELOG.md` if the change is user-facing) if
   your change affects usage, output, or accuracy.
4. Open a pull request with a short description of what changed and why.
