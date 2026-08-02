# ddc

Multiple sclerosis (MS) vs. control participant analysis project.

## Setup (Anaconda)

```bash
conda env create -f environment.yml
conda activate ddc-env
python -m ipykernel install --user --name ddc-env
```

## Structure

- `notebooks/` — analysis notebooks
- `data/` — participant data (raw + preprocessed)

## Notebook contents

1. Load data & environment setup
2. Demographics table
3. Age/gender-matched dataframe (matched to MS participants)
4. Predictor variable list (from preprocessed data)
5. Data exploration (distributions, missing values, interpolation)
6. Correlation between predictors (across + within group, 5 plots)
7. Elastic net & multinomial logistic regression notes
8. Elastic net model (controls dummy-coded to 0)
