# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-09-17

Initial release.

### Added

- `train.py`: trains an `SVC` (RBF kernel, `gamma=0.001`, `C=10`) classifier
  on scikit-learn's bundled `load_digits` dataset (1,797 samples of 8x8
  grayscale digit images), using an 80/20 train/test split with a fixed
  `random_state=42`. Prints the resulting test accuracy and saves the
  trained model plus the exact test split to `model.joblib`.
- `predict.py <index>`: loads `model.joblib` and predicts the digit at the
  given index of the saved test split, printing the predicted digit next
  to the ground-truth digit. Handles a missing model file and an
  out-of-range index with clear error messages instead of a raw traceback.
- `tests/test_digit_recognizer.py`: pytest suite covering model creation,
  accuracy above a 90% threshold, valid prediction ranges across single
  and multiple indices, out-of-range index handling, and overall
  prediction/ground-truth agreement.
- `model.joblib` committed directly to the repo (well under 1MB, since
  `load_digits` images are tiny) so the repo is immediately runnable
  without a training step.
- GitHub Actions CI (`.github/workflows/tests.yml`) running the test suite
  on Python 3.11 and 3.12.
- `LICENSE` (MIT) and initial `README.md` documenting usage, example
  output, project structure, and tech stack.

[0.1.0]: https://github.com/pastbologanesh0101/digit-recognizer/releases/tag/v0.1.0
