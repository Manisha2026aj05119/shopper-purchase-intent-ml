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
| Logistic Regression | I got the best result with C=0.1 and balanced class weights. Balancing pushed recall up from 0.36 to 0.74, but precision dropped to 0.50, so a lot of the predicted buyers were false alarms. Accuracy ended at 84.5%, which is almost the same as always predicting 'no purchase' (84.5%), so accuracy alone tells us very little here. The AUC of 0.895 is decent. The `PageValues` coefficient was by far the largest, so that one feature drives most of its decisions. It is quick and easy to read, but a straight-line boundary cannot capture everything in this data. |
| Decision Tree | This was my strongest model. With max_depth=6 and min_samples_leaf=20 it gave the best AUC (0.914), the best recall (0.82, only 67 of 382 buyers missed) and the best MCC (0.570). The default tree was a good lesson in overfitting: it scored 1.00 accuracy on the training data (depth 25, 803 leaves) but only an AUC of 0.736 on the test set, so I had to limit its depth. `PageValues` makes up 0.82 of the importance, followed by a November visit (0.06) and product-page duration (0.03). It is also easy to explain because you can read the rules. |
| KNN | With k=31 and distance weighting, KNN has the highest accuracy (0.873) and precision (0.74), which looks good at first. But its recall is only 0.28: it predicted 145 purchases and missed 275 real ones. Accuracy is misleading here because most sessions are 'no purchase'. 65 of the 75 inputs are one-hot columns, which probably makes the distances less meaningful, and KNN has no class_weight option, so I could not fix the imbalance that way. Its AUC (0.847) is second-lowest, and tuning actually reduced its F1 (0.406 vs 0.490 with defaults). |
| Naive Bayes | Gaussian Naive Bayes was the weakest model (AUC 0.807, MCC 0.384). It assumes the features are independent, but they are not: page counts and their durations are correlated (r = 0.60 to 0.86). A Gaussian is also a poor fit for features that are mostly zero or one-hot. With the default var_smoothing it predicted 'purchase' for almost everything (accuracy 0.27, recall 0.97), and setting var_smoothing=0.1 fixed that. Its one advantage is speed: it trained faster than all the others (0.03 s with defaults). |
| Support Vector Machine | The RBF SVM (C=4.0, gamma=0.01, balanced) came a close second. It has the best F1 (0.635) and is second on MCC (0.565) and AUC (0.902), with higher precision than the tree (0.55 vs 0.51). Balanced weights raised its recall from 0.47 to 0.75. The downside is cost: it was the slowest to tune (15 s of cross-validated search) and to score, and it is hard to interpret. |
| **Overall, Winner for your dataset?** | I pick the **Decision Tree**. It has the best AUC (0.914), recall (0.82) and MCC (0.570), and its F1 (0.633) is only about 0.002 behind the SVM's. The SVM beats it on accuracy (0.867 vs 0.852) and precision (0.55 vs 0.51), but the tree finds 30 more real purchases at the cost of 68 more false alarms. I think that is the right trade-off, because a missed buyer costs more than a wasted offer. The tree is also cheap to train (its cross-validated search took 1.1 s against 15 s for the SVM) and I can read its splits directly. KNN would have won on accuracy alone, but it missed 275 of 382 purchases, so accuracy is the wrong yardstick on this imbalanced data. |

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` in the app to reproduce the table above. To retrain, run `train_model.ipynb` after
`pip install -r requirements-dev.txt`.
