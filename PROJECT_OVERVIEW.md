# DDC Project Overview

## Project goal

This project examines whether movement-derived IMU features can distinguish between neurological groups, with a focus on comparing multiple sclerosis (MS) and control cohorts. Over the course of the work, the project evolved from exploratory preprocessing and feature engineering into a formal classification and permutation-testing workflow.

The analysis is built around raw movement data captured from smartwatch IMU recordings and uses engineered features such as tremor power, movement amplitude, spectral entropy, acceleration magnitude, jerk, and frequency-domain characteristics.

---

## Repository structure

- `README.md` — project introduction and setup notes
- `environment.yml` — Python environment and dependency list
- `data/` — raw and preprocessed participant datasets
- `notebooks/` — exploratory analysis, feature engineering, correlation studies, and classification notebooks
- `openspec/` — project planning and change-tracking artifacts

---

## Core technical stack

The project was run in a Python environment using:

- Python 3.12
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- pingouin

This combination supports the full pipeline: loading motion data, building predictors, performing EDA, running classification models, assessing performance, and computing permutation-based significance testing.

---

## Main analysis workflow

### 1. Data ingestion and setup

The project begins by loading the movement data from the preprocessed binary IMU files and mapping the participant IDs to the relevant diagnostic groups.

Key steps include:

- identifying the movement files
- extracting participant IDs
- stacking the IMU channels into a consistent array format
- defining the diagnostic groups and class labels
- building a participant-to-group lookup

### 2. Feature engineering from raw movement signals

The central methodological move was to generate movement-based predictors directly from the raw IMU data rather than relying on precomputed summaries.

The notebooks build features such as:

- Relative Tremor Power
- Harmonic Power Ratio
- Movement Amplitude
- Acceleration Norm
- Jerk
- Spectral Entropy
- Primary Frequency Component

These are derived from accelerometer-derived magnitude and frequency-domain behavior, using FFT-based analysis and signal summaries.

### 3. Model construction and validation

The project uses a supervised classification pipeline built around:

- StandardScaler
- LogisticRegression

Cross-validation is used with:

- StratifiedKFold
- cross_val_predict

This produces out-of-fold predictions, which are then used for:

- accuracy assessment
- classification reports
- confusion matrix plotting
- model interpretation

### 4. Permutation testing

A major recent focus was the permutation test used to estimate whether the observed classification accuracy significantly exceeds chance.

The workflow used:

- permutation_test_score from scikit-learn
- 10,000 permutations
- a null distribution of shuffled-label accuracies
- the observed score compared against this null distribution
- a p-value computed from the proportion of null accuracies as large as or larger than the observed accuracy

This was a key part of the last several days of work and was used to answer the question: is the observed model performance better than a random-label classifier?

---

## Notebooks in the repository

### Classification notebooks

- `classification_no_ms.ipynb` — main classification workflow without the MS-specific structure; used for the permutation-testing and confusion-matrix analysis discussed in the recent work.
- `classification.ipynb` — broader classification analysis.
- `classification_revised.ipynb` — revised version of the classification pipeline.
- `classification file.ipynb` — additional classification work or exploratory variant.

### Correlation notebooks

- `correlation_no_ms.ipynb` — correlation analysis without the missing-MS-specific setup.
- `correlation_all.ipynb` — correlation analysis across all available data.
- `predictors_correlation.ipynb` — correlation among predictors.
- `predictors_correlation_revised.ipynb` — revised predictor correlation analysis.

### Exploration notebooks

- `ddc_analysis.ipynb` — general project analysis notebook.
- `movement_research.ipynb` — movement-focused exploratory work.

---

## What you have been working on over the last several days

The work has been strongly centered on clarifying and validating the classification analysis, especially around the following questions:

1. Are the movement predictors being computed from raw IMU signals?
2. Is the classification accuracy meaningfully better than chance?
3. How should the null distribution be interpreted?
4. What is the correct way to compute the permutation p-value?
5. How should confusion-matrix diagnostics be placed in the notebook pipeline?
6. What does the histogram y-axis represent when using count versus density?
7. How can the notebook output be simplified and interpreted cleanly?

### Major decisions and clarifications

- The predictors are engineered from raw movement signals, not from a precomputed table.
- The plotting was switched to a count-based histogram to match the interpretation of the null distribution.
- The p-value should be computed as the proportion of shuffled accuracies as extreme as the observed accuracy.
- The confusion matrix belongs directly after `cross_val_predict`, because it should be built from the out-of-fold predictions.
- The permutation test was run with 10,000 permutations to generate a stable null distribution.

---

## The central results from the recent analysis

The most recent classification run produced the following key findings:

- Observed cross-validated accuracy: 0.48
- Permutation null mean: ~0.249
- Permutation p-value: ~0.0006

This indicates that the observed accuracy is substantially above chance and unlikely to arise under a random-label null model.

The confusion matrix also showed that the classes were not equally difficult:

- Essential Tremor showed relatively stronger classification performance.
- Parkinson’s and Atypical Parkinsonism were more often confused with one another.
- Healthy participants were also frequently confused with other classes.

This suggests some group separability, but also notable overlap between diagnostic categories.

---

## Overall interpretation

The project has moved from a data-preparation and feature-generation phase into a proper inferential analysis phase. The main narrative is that motion-derived features can capture clinically relevant differences across groups, and the recent permutation testing adds a formal assessment of whether the observed separability is statistically meaningful beyond chance.

In practical terms, the work has built a coherent pipeline that:

- extracts informative movement features from raw IMU data,
- fits a multigroup logistic model,
- evaluates performance with cross-validation,
- visualizes class confusion,
- tests significance through label permutation.

---

## Recommended next steps

1. Consolidate the strongest notebook(s) into a polished final report notebook.
2. Decide whether the final workflow should emphasize:
   - classification results,
   - permutation significance,
   - or both in a single publication-ready notebook.
3. Clean up duplicate or exploratory notebooks and keep only the final analytical versions.
4. Document the feature definitions in a methods section.
5. Add a concise interpretation of the confusion matrix and the p-value for the final write-up.

---

## Summary

This project is a movement-signal classification analysis using raw IMU-derived features and standard Python machine-learning workflows. Over the past several days, the work has centered on feature construction, cross-validated classification, confusion-matrix diagnostics, and permutation-based testing to assess significance. The project is now at the point where it has a coherent analytical pipeline and a strong result: the observed classification accuracy is clearly above chance, with a very small permutation p-value.
