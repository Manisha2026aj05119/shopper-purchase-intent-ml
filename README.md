# Shopper Purchase Intent Prediction

Author: Manisha (2026aj05119)

## a. Problem statement

An online store wants to know, while a visitor is still browsing, whether the session will end in a purchase.
This project treats that as a **binary classification** problem: given the behaviour recorded in one browsing
session (pages visited, time spent, bounce and exit rates, page value, month, visitor type and so on),
predict the `Revenue` flag, which is true when the session ended with a transaction.

Five classifiers are trained on the same dataset, evaluated with six metrics on a held-out test set, saved to
disk, and served through a Streamlit app that scores an uploaded test CSV without retraining.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | [UCI Machine Learning Repository, id 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (CC BY 4.0) |
| Instances | 12,330 sessions (minimum required: 500) |
| Input features | 17: 10 numeric and 7 categorical (minimum required: 12) |
| Target | `Revenue`, true if the session ended in a purchase |
| Class balance | 10,422 no purchase (84.53%) and 1,908 purchase (15.47%) |
| Missing values | None |
| Duplicate rows | 125, kept because the file has no session id |

| Feature group | Columns |
|---|---|
| Page counts | `Administrative`, `Informational`, `ProductRelated` |
| Page durations (seconds) | `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` |
| Analytics metrics | `BounceRates`, `ExitRates`, `PageValues` |
| Calendar | `SpecialDay`, `Month`, `Weekend` |
| Visitor and technical | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` |

**Preprocessing.** Numeric columns are standardised. The integer-coded columns `OperatingSystems`, `Browser`,
`Region` and `TrafficType` are labels, not quantities, so they are one-hot encoded along with `Month`,
`VisitorType` and `Weekend`. This gives 75 model inputs. The transformer lives inside each saved pipeline and
was fitted on the training split only, so the saved models accept the raw CSV columns directly.

**Split.** 80% train (9,864 rows) and 20% test (2,466 rows), stratified on `Revenue`, `random_state=42`.
The test split is saved as `test_data.csv`.

**Tuning.** Each model was tuned with `GridSearchCV` (stratified 5-fold, scored by ROC-AUC) on the training
split only. Class imbalance was handled with `class_weight="balanced"` where the model supports it.

## c. GitHub repository

https://github.com/Manisha2026aj05119/shopper-purchase-intent-ml

```
shopper-purchase-intent-ml/
|-- app.py                  Streamlit app
|-- train_model.ipynb       data exploration, training, tuning and evaluation
|-- requirements.txt        runtime dependencies (pinned)
|-- requirements-dev.txt    extras for running the notebook
|-- test_data.csv           held-out test set (2,466 rows, features plus Revenue)
|-- data/                   raw UCI dataset
|-- model/                  five saved pipelines (.joblib), registry and feature schema
|-- results/                metrics, confusion matrices, reports and plots
```

**Live app:** https://manisha2026aj05119-shopper-purchase-intent-ml.streamlit.app/

## d. Models used

All five models were trained on the same split and scored on the same 2,466 test sessions. The positive class
is a purchase.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8451 | 0.8949 | 0.5000 | 0.7382 | 0.5962 | 0.5194 |
| Decision Tree | 0.8516 | 0.9136 | 0.5130 | 0.8246 | 0.6325 | 0.5699 |
| KNN | 0.8731 | 0.8470 | 0.7379 | 0.2801 | 0.4061 | 0.4028 |
| Naive Bayes | 0.7948 | 0.8070 | 0.3980 | 0.6335 | 0.4889 | 0.3844 |
| Support Vector Machine | 0.8670 | 0.9023 | 0.5523 | 0.7461 | 0.6347 | 0.5650 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Cross-validation picked strong regularisation (C=0.1) with balanced class weights. Recall rose from 0.36 (defaults) to 0.74 and MCC from 0.459 to 0.519, at the cost of precision (0.50) and accuracy, which ends at 84.5% while always predicting 'no purchase' would score 84.5%. AUC 0.895 shows the linear boundary ranks sessions well; `PageValues` dominates the coefficients. A solid, fast and interpretable baseline, but the straight-line boundary limits it on this data. |
| Decision Tree | A shallow tree (max_depth=6, min_samples_leaf=20, balanced) is the strongest model: best AUC (0.914), best recall (0.82, only 67 of 382 purchases missed) and best MCC (0.570). The default unpruned tree memorises the training set (accuracy 1.00, depth 25, 803 leaves) but reaches only AUC 0.736 on the test set, so depth and leaf-size limits were essential. Splits are driven overwhelmingly by `PageValues` (importance 0.82); the next largest are a November visit (0.06) and product-page duration (0.03). The rules are easy to explain to a business user. |
| KNN | k=31 with distance weighting gives the highest accuracy (0.873) and precision (0.74) but the lowest recall (0.28): it flags only 145 sessions as purchases and misses 275 real ones. 65 of the 75 scaled + one-hot inputs are binary columns, which probably blur the distances, so neighbourhoods are noisy; KNN also has no class-weight option to correct the imbalance. Its AUC (0.847) is second-lowest, ahead only of Naive Bayes, and tuning for AUC actually lowered its F1 (0.406 vs 0.490 with defaults). |
| Naive Bayes | Gaussian NB is the weakest model (AUC 0.807, MCC 0.384). Its conditional-independence assumption is violated (page counts and their durations are correlated, r = 0.60 to 0.86) and a Gaussian is a poor fit for zero-inflated, right-skewed features and for one-hot columns. With the default var_smoothing it collapsed to predicting 'purchase' almost everywhere (accuracy 0.27, recall 0.97); var_smoothing=0.1 repairs it to a usable but still inferior model. Its one advantage is speed: it fits faster than every other model (0.04 s with defaults). |
| Support Vector Machine | The RBF SVM (C=4.0, gamma=0.01, balanced) is a close second: best F1 (0.635), MCC 0.565 and AUC 0.902, with the best precision/recall balance among the recall-oriented models (0.55 / 0.75). Balanced weights lifted recall from 0.47 to 0.75. It is the slowest model to tune (15 s of cross-validated search) and to score, and it offers no direct interpretability. |
| **Overall, Winner for your dataset?** | **Decision Tree** (tuned). It leads on the three metrics that matter under a 15 % positive rate: AUC 0.914, recall 0.82 and MCC 0.570, with F1 0.633, about 0.002 behind the best. The Support Vector Machine is a close second (MCC 0.565) and is ahead on accuracy (0.867 vs 0.852) and precision (0.55 vs 0.51), but the tree catches 30 more real purchases at the cost of 68 more false alarms, so it is the better pick when a missed buyer costs more than a wasted offer. The Support Vector Machine is also much slower to tune and score and is a black box, while the tree's cross-validated search took 1.1 s and its splits can be read directly. Accuracy would have crowned KNN, which misses 275 of 382 purchases, a reminder that accuracy is the wrong yardstick on imbalanced data. |

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app to reproduce the table above. To retrain, run `train_model.ipynb` after
`pip install -r requirements-dev.txt`.
