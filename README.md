# MLflow Credit Risk — Default Scoring

> Imbalanced classification · Credit default ranking · Lift and cumulative gains

## Business Problem

A lender must decide which loan applicants are likely to fall into serious delinquency (90+ days
past due) within two years. The decision the model informs is **approve, decline, or send for
manual underwriting review**, and — because the underwriting team is finite — **which applications
to review first**.

The cost of error is asymmetric. Approving a future defaulter (false negative) loses the
outstanding principal; declining a good applicant (false positive) loses the interest margin and a
customer relationship. Defaults are only ~6.7% of applicants, so accuracy is uninformative; the
model is judged on its ability to **rank risk** — lift and defaulters captured at a review capacity.

A score cutoff on a single variable (utilization, or any past-due count) captures part of the
signal but misses how income, debt ratio and the three delinquency histories combine, which the
model uses to separate borderline applicants.

## Dataset

Kaggle [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) (a
consent-gated competition; the prepared file is versioned in full as the reproducible source).

| Property | Value |
|----------|-------|
| Rows | 150,000 applicants |
| Target | `target` (1 = 90+ days past due within 2 years) |
| Positive rate | 6.7% (imbalanced) |
| Key features | revolving utilization, past-due history, debt ratio, income, age |

## Solution Strategy

1. **Acquisition** — the upstream competition is consent-gated, so the prepared dataset is versioned and loaded directly; column names were normalized to English.
2. **Leakage control** — all features describe the applicant at decision time; there is no post-outcome information.
3. **Feature engineering** — total past-due count, income per dependent and an income-missing flag, inside the model `Pipeline` so serving reuses the exact transform.
4. **Imbalance** — `class_weight="balanced"` inside the fitted folds; the metric is average precision (PR-AUC), appropriate for 6.7% positives.
5. **Model selection** — `StratifiedKFold` cross-validation compares a logistic baseline against histogram gradient boosting on average precision; the winner is tuned with `RandomizedSearchCV`.
6. **Evaluation** — ROC AUC and average precision on a stratified holdout, plus lift and defaulters captured at review capacities and ROC AUC by delinquency segment.

## Top Insights & Hypotheses

- **Recent serious delinquency dominates.** `PastDue90` is the strongest predictor (permutation importance 0.12); a single 90-day lapse sharply raises default odds.
- **Revolving utilization is the second lever** — applicants near their credit limit default far more often.
- **The three past-due histories are complementary**, not redundant; the engineered total sharpens the ranking.
- **The model discriminates best among clean applicants** (ROC AUC 0.83) and less well among those already delinquent (0.70), where most are high-risk and harder to separate — flagged in Next Steps.

## Engineered Features

| Feature | Formula | Business signal |
|---------|---------|-----------------|
| TotalPastDue | `PastDue30_59 + PastDue60_89 + PastDue90` | Overall delinquency intensity across severities. |
| IncomePerDependent | `MonthlyIncome / (Dependents + 1)` | Disposable income proxy after household load. |
| MonthlyIncomeMissing | `1 if MonthlyIncome is null else 0` | Missing income is itself informative of risk. |

## Model

A histogram gradient boosting classifier (selected by cross-validation, tuned with randomized
search) inside a `Pipeline` that owns the engineering and scaling. The logistic baseline sets the bar.

| Model | CV avg precision | Holdout ROC AUC | Holdout avg precision |
|-------|-----------------:|----------------:|----------------------:|
| Logistic baseline | 0.293 | 0.802 | 0.323 |
| **Hist gradient boosting (final)** | **0.391** | **0.870** | **0.408** |

## Business Results

Ranking the holdout by default score and reviewing top-N applicants:

| Reviewed | Precision@k | Lift vs random | Defaulters captured |
|----------|------------:|---------------:|--------------------:|
| 1,000 | 54.9% | 8.21x | 27.4% |
| 5,000 | 27.5% | 4.12x | 68.6% |
| 10,000 | 17.2% | 2.58x | 85.9% |

Reviewing the top 5,000 highest-risk applicants flags **69% of future defaulters at 4.1x** the
efficiency of random review; the top 1,000 are over **8x** more likely to default than average, so
the underwriting team can concentrate scrutiny where it pays off most.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/mlflow-credit-risk.git
   cd mlflow-credit-risk
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data** — the prepared dataset is versioned under `data/sample/`; no Kaggle download is needed because the upstream competition is consent-gated.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [huggingface.co/spaces/pmusachio/mlflow-credit-risk](https://huggingface.co/spaces/pmusachio/mlflow-credit-risk) — score an applicant and explore the underwriting capacity view.

## Next Steps

- The original project framed reproducibility around an MLflow tracking server; this refactor replaces it with a deterministic pipeline plus a versioned `model_card.json`, which delivers the same auditability without the server overhead for a single production model. Re-introducing MLflow would pay off only with many concurrent experiments.
- Calibrate probabilities and attach loss-given-default so decisions maximize expected portfolio value rather than raw default probability; deferred until exposure data is available.
- Improve separation among already-delinquent applicants with bureau and trade-line features the current snapshot lacks.
