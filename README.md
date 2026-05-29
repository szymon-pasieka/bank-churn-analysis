# Bank Customer Churn Analysis

Descriptive and predictive analysis of retail-bank customer churn using the
[Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
— ~10,000 customers of a retail bank across France, Spain, and Germany, with
demographics, account features, and a churn flag (`Exited`). The analysis answers
two questions a junior analyst on a retention team might be asked.

## Key findings

_Populated as the analysis is completed — see the notebooks for current state._

## Questions explored

1. **Which customer segments are most likely to churn, and what are they doing
   differently?** Churn rates by geography, age, gender, product holdings, balance,
   and account activity, plus the highest-risk combined segments. (`01_exploration.ipynb`)
2. **What factors most strongly predict churn, holding other factors constant?**
   A logistic regression model with interpreted coefficients (odds ratios),
   evaluated as a classifier. (`02_modeling.ipynb`)

## Data

A single table, one row per customer:

- `CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary` — numeric attributes
- `Geography` (France / Spain / Germany), `Gender` — categorical demographics
- `NumOfProducts` (1–4), `HasCrCard` (0/1), `IsActiveMember` (0/1) — products and engagement
- `Exited` (0/1) — the target: whether the customer left the bank
- `RowNumber`, `CustomerId`, `Surname` — identifiers, dropped before analysis

## Methods

- Loading, cleaning, and sanity-checking the dataset
- Segmentation with `groupby` + `.agg`, and bucket analysis of continuous variables with `pd.cut`
- Feature engineering: categorical encoding, deliberate handling of a non-monotonic predictor, derived features
- Logistic regression with `statsmodels` — coefficient significance, confidence intervals, odds-ratio interpretation
- Classification evaluation: stratified train/test split, confusion matrix, precision / recall / F1, ROC-AUC, threshold selection under class imbalance
- Visualization: bar charts and grouped comparisons against the baseline churn rate

## Repo structure

```
.
├── 01_exploration.ipynb   # Descriptive segmentation — who churns
├── 02_modeling.ipynb      # Logistic regression — what predicts churn
├── README.md              # You are here
├── requirements.txt       # Pinned package versions
├── .gitignore
└── data/                  # Not committed — download from Kaggle (see below)
```

## How to run

1. Download `Churn_Modelling.csv` from
   [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) and
   place it in a `data/` folder at the project root.
2. Set up the environment:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

3. Run `01_exploration.ipynb`, then `02_modeling.ipynb`, top to bottom.

## Tools

- Python 3.12
- pandas, numpy
- matplotlib, seaborn
- statsmodels (logistic regression + inference)
- scikit-learn (train/test split, evaluation metrics)
- Jupyter / VS Code

## Caveats

- The Churn Modelling dataset is a widely used teaching dataset with no documented
  real-world provenance; findings demonstrate method rather than production bank intelligence.
- The analysis is observational — the model identifies predictors associated with churn, not causes.
- The data is a single cross-sectional snapshot with no time dimension, so churn timing and the
  behaviour leading up to churn cannot be observed.

## Possible extensions

- **Cost-based threshold selection** — choose the probability cutoff that minimises expected cost
  given the price of a retention offer versus the value of a lost customer, rather than a default 0.5.
- **Non-linear model comparison** — fit a tree-based model (random forest / gradient boosting) to
  check for interactions and non-linearities the logistic model assumes away, and compare ROC-AUC.
- **Class-imbalance handling** — compare the baseline against class-weighted or resampled (SMOTE)
  variants and report the effect on recall.
