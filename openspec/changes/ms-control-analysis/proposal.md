## Why

The ddc project needs a structured, reproducible analysis pipeline comparing Multiple Sclerosis (MS) patients with healthy controls, but the existing notebook is only a scaffold with no agreed-upon design, specifications, or task breakdown. We need a shared contract — captured in OpenSpec — describing exactly what the analysis must do so future work can be built, validated, and extended incrementally instead of drifting in chat history.

## What Changes

- Define the end-to-end analysis pipeline as a specification: demographics table, age/gender matching, predictor selection, data exploration, correlation analysis, and elastic net / multinomial logistic regression.
- Establish the MS-control analysis capability in `openspec/specs/` with concrete requirements and scenarios.
- Provide a design documenting the technical approach (data flow, preprocessing, matching algorithm, model configuration).
- Provide a task list that can be implemented incrementally by an AI assistant or human.

## Capabilities

### New Capabilities
- `ms-control-analysis`: The reproducible pipeline for comparing MS vs. Control participant data — from raw/preprocessed participant data through demographics, age/gender matching, data exploration, correlation analysis, and regularized regression modelling.

### Modified Capabilities
<!-- No existing specs yet; this is the first capability. -->

## Impact

- **Code**: `notebooks/ddc_analysis.ipynb` will be updated to implement the pipeline; helper functions may be extracted to a shared module.
- **Data**: expects participant data in `data/` (raw + preprocessed); currently `data/demo_participants.csv` used for development.
- **Dependencies**: Python 3.12, `ddc-env` conda environment (pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, pingouin).
- **Process**: enables OpenSpec workflow (`/opsx-apply`, `/opsx-archive`) for this and future changes.
