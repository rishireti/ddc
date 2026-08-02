## Context

The ddc project currently has a scaffolded notebook (`notebooks/ddc_analysis.ipynb`) and demo participant data (`data/demo_participants.csv`, columns: `participant_id`, `group`, `age`, `gender`) in the `ddc-env` conda environment. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- A reproducible MS vs. Control analysis pipeline implemented primarily in a Jupyter notebook, with analysis code testable end-to-end.
- Deterministic, seeded results across every stage.

**Non-Goals:**
- Collecting or preprocessing real participant data (expected as input).
- Deployment or productionization; results are reported in the notebook.
- Building a full application or package distribution.

## Decisions

- **Notebook-first, functions extracted when reused.** Keep the pipeline in `notebooks/ddc_analysis.ipynb`; move reusable logic (matching, missing-data handling, elastic net runner) into a shared module (e.g. `src/ddc/analysis.py`) so cells stay readable and logic is unit-testable.
- **Matching algorithm — greedy nearest-neighbour on gender + age.** Controls are matched to MS participants (MS is the reference because it is the smallest group): for each MS participant, select the available Control of the same gender minimizing absolute age difference, then remove that Control from the pool. Alternatives considered: 1:1 optimal matching (networkx / `MatchIt`-style, heavier) and exact age matching (infeasible when bins are sparse). Greedy nearest-neighbour is simple, deterministic with a fixed seed, and satisfies the spec.
- **Missing data — report-then-impute with interpolation preference.** Explore missingness per predictor; prefer linear interpolation (`df.interpolate()`) for time-ordered numeric sequences, otherwise median/imputation for independent columns, and drop only when a column is mostly missing. Explicitly documented in the notebook so the choice is auditable.
- **Correlations — across-group plus per-group heatmaps.** Use `seaborn.clustermap` for the across-group matrix and `seaborn.heatmap` per group; total plot count is 1 + number of groups (spec requires exactly this).
- **Modelling — elastic net via scikit-learn.** For a binary outcome (MS=1, Control=0), use `LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=...)` inside a `StandardScaler` pipeline, tuning `l1_ratio` and `C` with `GridSearchCV` (scoring `roc_auc`), with a stratified train/test split at a fixed seed. Alternative considered: `ElasticNet` for regression — rejected because the outcome is categorical. Multinomial logistic regression is documented as an option if the outcome becomes multi-class (e.g., MS subtypes).
- **Controls dummy-coded to 0.** The outcome `y = (group == "MS").astype(int)` guarantees Control = 0, MS = 1.
- **Reproducibility.** Fixed `random_state` on all stochastic steps (matching, split, CV). Record library versions in the notebook's final cell.

## Risks / Trade-offs

- Greedy matching is not globally optimal → [Mitigation] document the algorithm; acceptable for typical cohort sizes and swap to optimal matching only if matching quality is challenged.
- Small samples make CV unstable → [Mitigation] report CV ± spread, use stratified splits, and interpret coefficients cautiously.
- Notebook outputs can drift from code → [Mitigation] execute the notebook end-to-end (`jupyter nbconvert --execute`) as part of verification.
- Real data may have heavy missingness → [Mitigation] exploration stage reports missingness before imputation decisions are made.

## Open Questions

- Whether the outcome will stay binary or become multi-class (MS subtypes) — deferred; the design documents both paths and the binary path is implemented first.
