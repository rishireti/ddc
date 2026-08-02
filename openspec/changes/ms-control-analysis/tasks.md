## 1. Setup & Data Loading

- [ ] 1.1 Ensure `ddc-env` conda environment is active and kernel registered in Jupyter
- [ ] 1.2 Load participant data (raw + preprocessed) into the notebook from `data/`
- [ ] 1.3 Define shared constants (group labels, seed, output paths) at the top of the notebook

## 2. Demographics Table

- [ ] 2.1 Build demographics summary table grouped by group (count, age mean/sd, gender counts and percentages)
- [ ] 2.2 Verify table renders correctly for both MS and Control groups

## 3. Age- and Gender-Matched Dataframe

- [ ] 3.1 Implement `match_controls()` — greedy nearest-neighbour match of Controls to MS on gender + closest age, deterministic with a fixed seed
- [ ] 3.2 Apply matching to build the analysis dataframe and report matched counts per group
- [ ] 3.3 Check the matched dataframe has no duplicated Controls and preserves MS participants

## 4. Predictor Variable Selection

- [ ] 4.1 Document the predictor variable list sourced from preprocessed data (exclude ids, group, outcome)
- [ ] 4.2 Build the numeric feature matrix `X` and confirm expected shape

## 5. Data Exploration

- [ ] 5.1 Plot/summarize the distribution of each predictor separately by group
- [ ] 5.2 Report missing values per predictor and decide a handling strategy (drop / impute / interpolate)
- [ ] 5.3 Apply the documented missing-data strategy and confirm no NaNs remain in the feature matrix

## 6. Correlation Analysis

- [ ] 6.1 Produce the across-group correlation plot (all participants)
- [ ] 6.2 Produce one correlation plot per group
- [ ] 6.3 Verify the total plot count is exactly one across-group plot plus one per group

## 7. Model Research Notes

- [ ] 7.1 Write up research notes on elastic net (loss, L1/L2 penalties, pros/cons, tuning)
- [ ] 7.2 Write up research notes on multinomial logistic regression and when it applies (multi-class outcome)

## 8. Elastic Net Model

- [ ] 8.1 Create outcome `y` with MS = 1 and Controls dummy-coded to 0
- [ ] 8.2 Build the `StandardScaler` + elastic-net `LogisticRegression` pipeline (solver `saga`, `l1_ratio` parameter)
- [ ] 8.3 Set up stratified train/test split with a fixed random seed
- [ ] 8.4 Tune `l1_ratio` and `C` via `GridSearchCV` (scoring `roc_auc`)
- [ ] 8.5 Report best parameters, CV score, held-out score, and per-predictor coefficients

## 9. Verification

- [ ] 9.1 Execute the notebook end-to-end (`jupyter nbconvert --execute`) without errors
- [ ] 9.2 Record library/Python versions in the final session-info cell
- [ ] 9.3 Confirm outputs (tables and plots) are present and reproducible across runs
