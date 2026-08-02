## Purpose

Defines the reproducible pipeline that compares Multiple Sclerosis (MS) participants against Controls on demographics, matching, data quality, correlations, and regularized regression modelling.

## ADDED Requirements

### Requirement: Demographics table
The pipeline SHALL produce a demographics summary table grouped by group (MS / Control) that includes participant counts and age/gender summary statistics.

#### Scenario: Demographics table generated
- **WHEN** the analysis runs on the participant data
- **THEN** the output contains one row per group with participant count, mean and standard deviation of age, and male/female counts and percentages

### Requirement: Age- and gender-matched dataframe
The pipeline SHALL produce an analysis dataframe where Controls are matched to MS participants on gender, selecting the control with the closest age (absolute difference); when multiple controls tie, one SHALL be selected deterministically. The MS group SHALL be the reference because it is the smallest sample.

#### Scenario: Control matched to MS participant
- **WHEN** a matched dataframe is created from MS and Control participants
- **THEN** every retained Control has the same gender as its matched MS participant and the smallest possible absolute age difference among available candidates

#### Scenario: Deterministic tie-breaking
- **WHEN** two or more candidate Controls are equally close in age to the same MS participant
- **THEN** the same single candidate is selected across repeated runs using a fixed random seed

### Requirement: Predictor variables from preprocessed data
The pipeline SHALL select predictor variables from the preprocessed participant data, excluding identifiers and the outcome/group variable, and SHALL document the chosen predictors.

#### Scenario: Predictor list assembled
- **WHEN** the predictor list is built from preprocessed data
- **THEN** the output is a numeric feature matrix containing only the documented predictor columns with the expected number of rows and columns

### Requirement: Data exploration
The pipeline SHALL examine the distribution of every predictor, report missing values, and propose/apply a handling strategy (drop, impute, or interpolation) for any missing data.

#### Scenario: Distributions inspected
- **WHEN** the exploration step runs
- **THEN** each predictor is plotted or summarized (histogram/summary statistics) separately by group

#### Scenario: Missing values handled
- **WHEN** any predictor contains missing values
- **THEN** the missingness is reported and a documented strategy (drop, impute, or interpolate) is applied before modelling

### Requirement: Correlation analysis
The pipeline SHALL produce correlation matrices between predictors across all participants (across group) and within each group, yielding exactly one across-group plot plus one plot per group.

#### Scenario: Across-group correlation plot
- **WHEN** correlation is computed across all participants
- **THEN** one correlation heatmap/clustermap is produced using all participants together

#### Scenario: Within-group correlation plots
- **WHEN** correlation is computed within groups
- **THEN** exactly one correlation plot is produced per group, and the total plot count equals one across-group plot plus the number of distinct groups

### Requirement: Elastic net / multinomial logistic regression
The pipeline SHALL run a regularized regression where the outcome is binary (MS = 1, Control = 0) and Controls are dummy-coded to zero. Features SHALL be standardized and hyperparameters (alpha / l1_ratio or C) tuned via cross-validation.

#### Scenario: Model trained with controls coded to zero
- **WHEN** the elastic net model is fitted on predictors X and outcome y
- **THEN** y equals 1 for MS participants, 0 for Controls, and the fit reports cross-validated and held-out performance with the resulting coefficients per predictor

#### Scenario: Reproducible run
- **WHEN** the modelling step is executed repeatedly
- **THEN** identical results are produced because feature splitting and cross-validation use a fixed random seed
