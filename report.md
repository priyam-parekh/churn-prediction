**Predictive Modeling to Determine the Core Drivers of Churn and Retention in Subscription Services**  
**Author**: Priyam Parekh  
**Date**: Dec 12, 2025

## 1. Problem description

Churn is the rate at which customers stop using a product or cancel a subscription. For subscription businesses, churn is directly tied to revenue and growth, and it is often cheaper to retain an existing customer than to acquire a new one. Because of this, being able to predict churn before it happens is valuable: if a company can identify customers who are at high risk of leaving, it can take targeted retention actions (for example, proactive outreach, discounts, customer support interventions, or product improvements) while the customer is still active.

This project focuses on the following research questions. First, what customer attributes most strongly predict churn? Second, which attributes instead indicate long-term retention for subscription-based companies? Rather than only building a classifier, my goal is to build a model that is interpretable enough to support an analysis of the drivers of churn and retention.

Formally, the machine learning problem is supervised binary classification where the target variable is `Churn` (`1 = churned`, `0 = retained`). The core deliverables are (1) a working end-to-end machine learning pipeline for predicting churn and (2) a ranked set of features that explain which customer behaviors and account attributes are most connected to churn risk.

## 2. Dataset description and preparation

### 2.1 Sources

The dataset used in this project is from Kaggle and is titled **“Predictive Analytics for Customer Churn”**. In this repository, it is stored locally in the `customer-churn-dataset/` folder as three files: `train.csv`, `test.csv`, and `data_descriptions.csv`.

### 2.2 Structure and properties

The training split contains 243,787 rows and 21 columns (20 features plus the `Churn` label). The Kaggle test split contains 104,480 rows and 20 columns; importantly, it does not include `Churn` labels. Because of this, the Kaggle test split cannot be used as a fully labeled hold-out test set for reporting final generalization error. Instead, my primary generalization estimate comes from a validation split carved out of `train.csv`, and I also use stratified k-fold cross-validation to check stability.

The dataset is imbalanced. In the training data, `Churn = 1` occurs 44,182 times (18.12%), while `Churn = 0` occurs 199,605 times (81.88%). This imbalance is an important reason to emphasize metrics like AUC-ROC and F1 score rather than relying only on accuracy.

The feature set covers multiple aspects of a customer’s relationship with a subscription product. It includes account attributes such as `AccountAge` and `Gender`, billing and plan information like `SubscriptionType`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`, engagement features such as `ViewingHoursPerWeek`, `AverageViewingDuration`, `ContentDownloadsPerMonth`, `WatchlistSize`, `ContentType`, and `GenrePreference`, and support/experience features like `SupportTicketsPerMonth` and `UserRating`. It also includes binary settings/flags like `PaperlessBilling`, `ParentalControl`, `SubtitlesEnabled`, `MultiDeviceAccess`, and `DeviceRegistered`.

### 2.3 Data cleaning

All preprocessing is implemented in `src/data_preprocessing.py`. The first major cleaning step is handling missing values so that downstream models can be trained reliably. For numerical features, I use median imputation via `SimpleImputer(strategy='median')`. I chose the median because it is robust to outliers, which can be common in subscription/billing data. For categorical and binary features, I use most-frequent imputation via `SimpleImputer(strategy='most_frequent')` so that missing values are replaced with a plausible category rather than introducing a new artificial category.

Another practical issue is that categorical levels can appear in non-training splits that were not present in the data used to fit encoders. To avoid runtime failures, the pipeline maps unseen categorical values in non-training splits to a safe default category prior to applying the learned label encoding.

### 2.4 Feature engineering and dimensionality reduction

A major goal of this project is to connect raw dataset features to churn and retention mechanisms. For that reason, I implemented feature engineering in `ChurnDataPreprocessor.create_feature_engineering()` to create additional features that represent intensity and ratios, which are often more predictive than raw counts alone. The engineered features include:

- `TotalChargesPerMonth = TotalCharges / (AccountAge + 1)`
- `MonthlyChargesRatio = MonthlyCharges / TotalChargesPerMonth`
- `ViewingHoursPerMonth = ViewingHoursPerWeek * 4.33`
- `ViewingIntensity = ViewingHoursPerMonth / (AccountAge + 1)`
- `SupportIntensity = SupportTicketsPerMonth / (AccountAge + 1)`
- `ContentEngagement = ContentDownloadsPerMonth / (ViewingHoursPerWeek + 1)`
- `LowRating = 1(UserRating < 3)`
- `HighSupportUsage = 1(SupportTicketsPerMonth > 75th percentile)`

These features are designed to reflect real-world behaviors: for example, engagement intensity can capture declining usage patterns, and support intensity can indicate friction or dissatisfaction. I did not apply dimensionality reduction such as PCA because (1) gradient boosted tree models work well with a moderately sized feature set, (2) interpretability was a priority, and (3) PCA would make it harder to explain which real-world behaviors correspond to churn.

### 2.5 Categorical encoding and scaling

Categorical and binary features are converted to numeric form using label encoding (`sklearn.preprocessing.LabelEncoder`). This choice is primarily a pragmatic one: label encoding is simple and memory-efficient compared to one-hot encoding, especially when there are many categories. While one-hot encoding can sometimes be more appropriate for linear models, tree-based models can still perform well with integer-coded categories, and this approach kept the pipeline straightforward.

After splitting the data, I apply standardization using `StandardScaler`. Feature scaling is not strictly necessary for tree-based methods, but it makes engineered features more comparable in magnitude and keeps the pipeline flexible if other model families are tested later.

### 2.6 Justification

Overall, the preprocessing and feature engineering choices were made to create a robust and reproducible pipeline that still supports interpretability. Median/mode imputation prevents training failures from missing values without introducing overly complex assumptions. The feature engineering step explicitly targets churn-related mechanisms such as engagement and support burden. Finally, label encoding and a simple scaling procedure keep the code readable and the feature space manageable.

## 3. Machine learning methodology

### 3.1 Representation (hypothesis class)

The primary model family used in this project is **Gradient Boosted Decision Trees (GBDT)**. I trained and compared two popular GBDT implementations: XGBoost (`xgboost.XGBClassifier`) and LightGBM (`lightgbm.LGBMClassifier`). I chose GBDT models because churn behavior is unlikely to be purely linear; instead, churn often depends on non-linear thresholds and interactions (for example, high monthly charges may only predict churn when engagement drops, or support tickets may only matter beyond a certain point). GBDT models are well-known for handling these non-linear relationships and interactions effectively, and they also provide built-in feature importance measures that directly support interpretability.

### 3.2 Loss function

Because churn prediction is a binary classification task, both models are trained with logistic-style objectives. In XGBoost, this is specified as `objective='binary:logistic'`, and in LightGBM it is specified as `objective='binary'`. Both correspond to logistic loss (binary cross-entropy), which is appropriate for learning class probabilities in binary classification.

### 3.3 Optimization

GBDT optimization works through additive modeling: the model builds an ensemble of decision trees sequentially, and each new tree attempts to correct mistakes made by the current ensemble. In practice, the optimization behavior is controlled by hyperparameters such as the number of trees (`n_estimators`) and the learning rate (`learning_rate`). I used a regularized parameter configuration with `max_depth = 4`, `learning_rate = 0.1`, `n_estimators = 100`, `subsample = 0.8`, and `colsample_bytree = 0.8`. These parameters reduce overfitting by limiting tree complexity and introducing randomness through row and feature sampling.

The code also supports optional hyperparameter optimization using Optuna (TPE sampler). Optuna can search over depth, learning rate, number of estimators, and sampling rates. In the reported run, Optuna tuning was enabled with `--n-trials 20` for both XGBoost and LightGBM.

## 4. Evaluation of methods

The evaluation design is meant to be realistic for a churn prediction setting with imbalanced labels. I report AUC-ROC as the primary metric because it measures ranking quality across thresholds and is less sensitive to class imbalance than accuracy. I also compute accuracy, precision, recall, and F1 score so that the results are interpretable from multiple perspectives (for example, precision/recall tradeoffs matter for deciding how many at-risk customers a company should target).

To evaluate performance, I split the labeled training data (`train.csv`) into an 80/20 train/validation split with stratification so that the churn rate is preserved in both splits. In addition, I run stratified 5-fold cross-validation (K=5) using AUC-ROC to assess stability and reduce the chance that results are overly dependent on a single random split.

The reported quantitative results below correspond to a run with the full dataset and `random_state = 42`.

**Cross-validation during Optuna tuning (5-fold CV AUC-ROC on the training split):**

| Model    | Best CV AUC-ROC (mean) |
| -------- | ---------------------: |
| XGBoost  |                 0.7464 |
| LightGBM |                 0.7459 |

**Hold-out validation (default 0.5 threshold for F1):**

| Model    | Val AUC | Val F1 |
| -------- | ------: | -----: |
| XGBoost  |  0.7524 | 0.1888 |
| LightGBM |  0.7516 | 0.1874 |

These results show that both tuned models achieve moderate discrimination on the validation set (AUC around 0.75). The pipeline also raised a train/validation distribution mismatch warning, which suggests some shift between splits. The low F1 values reflect the default 0.5 classification threshold on an imbalanced dataset; in practice, the decision threshold should be tuned to the business objective (precision vs. recall).

## 5. Decision and adaptation

I trained XGBoost and LightGBM under the same preprocessing pipeline and evaluation setup so that the comparison would be fair. Based on validation performance after Optuna tuning, I selected XGBoost as the final model because it achieved the best validation AUC-ROC (0.7524), narrowly outperforming LightGBM (0.7516).

During development, I made several refinements to improve both model quality and interpretability. I added explicit regularization controls such as subsampling and feature sampling (`subsample` and `colsample_bytree`) and restricted tree depth (`max_depth`) to reduce variance. I also added feature engineering focused on churn and retention mechanisms (for example, support intensity and engagement intensity) so that the model could more directly learn patterns related to customer behavior.

In future work, I would enable Optuna tuning more aggressively to explore the bias-variance tradeoff and potentially reduce overfitting. I would also consider imbalance-aware methods such as class weights or threshold calibration, since business decisions about retention campaigns often depend more on recall and precision than on a default 0.5 probability threshold.

## 6. Implementation

### 6.1 Code structure

All project code lives in the `src/` directory, and the code is organized into modules that correspond to pipeline stages. The main orchestration happens in `src/main.py` (class `ChurnPredictionPipeline`). Preprocessing is implemented in `src/data_preprocessing.py`. Model training is implemented in `src/gbdt_models.py`. Validation and quality checks are implemented in `src/model_validation.py`. Evaluation metrics and model selection utilities are implemented in `src/model_evaluation.py`. Feature importance extraction and plotting are implemented in `src/feature_importance.py`, and exploratory plots are implemented in `src/churn_visualizations.py`. Helper functions for loading data and saving plots are in `src/utils.py`.

### 6.2 Main components and execution flow

The pipeline is designed to be runnable end-to-end from a single entry point. When `src/main.py` is executed, it loads the dataset, runs basic exploratory plots, applies preprocessing and feature engineering, splits the data into training and validation sets, trains two GBDT models, validates them using cross-validation and data-quality checks, evaluates them using multiple metrics, and finally creates feature importance and other visualizations.

To run the pipeline from the repository root:

```bash
# Use subset (default, ~2-5 minutes, good for testing)
python src/main.py

# Full dataset + Optuna tuning (reported run)
python src/main.py --use-full-dataset --optimize-hyperparameters --n-trials 20
```

### 6.3 Reproducibility

To support reproducibility, random seeds are fixed at 42 for data splitting and Optuna’s sampler (when enabled). Dependencies are listed in `requirements.txt`. The pipeline supports two modes:

- **Subset mode (default)**: Uses 10K training samples for faster iteration during development (~2-5 minutes).
- **Full dataset mode**: Use the `--use-full-dataset` flag to use all ~244K training samples and ~104K test samples. This provides the best model performance but takes longer; with Optuna enabled (`--n-trials 20`), the end-to-end run took ~13 minutes on my machine (runtime will vary).

The report results were produced with the full dataset (using `--use-full-dataset`) and Optuna hyperparameter optimization enabled (`--optimize-hyperparameters --n-trials 20`).

### 6.4 Use of external resources

This project uses standard Python data science libraries including pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, matplotlib, and seaborn. The dataset is sourced from Kaggle. All pipeline and modeling code under `src/` was written for this project using standard library APIs.

## 7. Final results and analysis

### 7.1 Validation and testing

Because Kaggle’s `test.csv` does not include `Churn` labels, I cannot report a true labeled test-set performance using the provided test split. Instead, I treat the hold-out validation metrics as the primary estimate of generalization error and use stratified cross-validation to validate that the model’s performance is reasonably stable across different folds.

### 7.2 Error reporting (train vs validation)

For the selected XGBoost model, the validation AUC-ROC is 0.7524 and the validation F1 score (at a 0.5 threshold) is 0.1888. An AUC around 0.75 indicates moderate discrimination. The low F1 value is expected under class imbalance when using a fixed 0.5 threshold; adjusting the decision threshold can substantially change precision/recall/F1 while leaving AUC largely unchanged. The pipeline also flagged a train/validation distribution mismatch, suggesting some shift between splits.

### 7.3 Generalization analysis

Using Optuna tuning, the best 5-fold CV AUC on the training split was ~0.746 and the hold-out validation AUC was ~0.752, suggesting performance is reasonably stable at a moderate level, though there is still room for improvement.

### 7.4 Visualizations

The pipeline saves several visualizations to the `results/` directory. These plots are used to support both exploratory analysis and model interpretation. The saved plots include `churn_correlation.png` (top numerical features correlated with churn), `churn_rate_by_category.png` (churn rates for selected categorical features), `feature_importance.png` (top features by GBDT feature importance), and `churn_predictors_vs_retention_indicators.png` (a side-by-side comparison of high-importance vs low-importance features).

### 7.5 Key drivers and interpretation

The final model’s feature importance provides a direct answer to the core research question about churn drivers. For the selected XGBoost model, the top churn drivers are:

1. MonthlyChargesRatio
2. AccountAge
3. SupportIntensity
4. AverageViewingDuration
5. ContentDownloadsPerMonth
6. ViewingHoursPerWeek
7. MonthlyCharges
8. ViewingHoursPerMonth
9. SubscriptionType
10. HighSupportUsage

A clear pattern emerges from this ranking. Engagement metrics (viewing duration, viewing hours, and downloads) are strong predictors; this aligns with the intuition that when customers stop using a service, they are more likely to churn soon after. Billing intensity and pricing-related variables (monthly charges and charge ratios) also appear as meaningful drivers, which suggests that higher prices may increase churn risk, especially when engagement is not high enough to justify the cost. Finally, satisfaction and support signals (user rating and support intensity) also matter, indicating that product quality and customer experience likely play a role.

To connect model interpretation with simpler exploratory statistics, I also compare model feature importance with correlation-based rankings for numerical features. The overlap between the model’s top features and the correlation ranking was moderate (5/10 overlap in the top-10 lists), which suggests that the model is leveraging both linear effects (captured by correlation) and non-linear relationships or interactions (captured by GBDT).

### 7.6 Reflection

Overall, this project achieved its main goal: building an interpretable churn classifier and producing a ranked list of churn drivers supported by exploratory visualizations. The main limitations are the observed generalization gap and the lack of labels in Kaggle’s provided test split, which prevents a true final test evaluation. If I continued this project, I would focus on stronger regularization and tuning (through Optuna), threshold calibration for better business-aligned precision/recall tradeoffs, and constructing a labeled hold-out test split from the training data so that final performance can be reported in a more standard way.

## Citations

- Kaggle dataset: “Predictive Analytics for Customer Churn” — `https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset`
- Libraries: pandas, scikit-learn, xgboost, lightgbm, optuna, matplotlib, seaborn
