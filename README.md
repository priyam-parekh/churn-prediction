# Customer Churn Prediction 

Predicting whether a subscription customer will churn or stay, and which features matter most for that prediction.

I chose this because it ties CS to something practical—for subscription businesses, losing customers directly hurts growth, and spotting who might leave early gives a chance to intervene. Churn isn’t driven by one thing; it’s billing, usage, support, demographics, etc. So the goal isn’t only to build a classifier but to see which levers actually move the needle.

**What the pipeline does:** Binary classification (Churn = 1 left, Churn = 0 stayed). I use gradient boosted trees—XGBoost and LightGBM—since they handle mixed feature types and give interpretable importance scores. Outputs are evaluation metrics (AUC-ROC, F1, etc.) and plots in `results/`.

**Data:** Kaggle dataset “Predictive Analytics for Customer Churn”  
https://www.kaggle.com/datasets/safrin03/predictive-analytics-for-customer-churn-dataset

Files used: `customer-churn-dataset/train.csv`, `test.csv`, and `data_descriptions.csv`. Note: Kaggle’s test set doesn’t have Churn labels, so the main performance numbers come from the validation split and cross-validation.

**Layout:**  
`src/main.py` runs the full pipeline. Preprocessing (missing values, encoding, feature engineering, scaling) lives in `data_preprocessing.py`. `gbdt_models.py` trains XGBoost and LightGBM and can run Optuna tuning. `model_validation.py` does stratified K-fold CV and basic data-quality checks. `model_evaluation.py` computes metrics and picks the best model. Feature importance and EDA plots are in `feature_importance.py` and `churn_visualizations.py`. `utils.py` has data loading and plot-saving helpers. Data sits in `customer-churn-dataset/`, figures go to `results/`.

**Setup:**  
From the repo root, create a venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run:**  
From repo root:

```bash
python src/main.py
```

By default it uses 10K training samples so runs finish in a few minutes. For full data (~244K train, ~104K test):

```bash
python src/main.py --use-full-dataset
```

You can also pass `--sample-size 5000` for a smaller run, or turn on Optuna with `--optimize-hyperparameters --n-trials 20`. To tune for accuracy instead of AUC (adds class weighting and threshold tuning), use `--optimize-for-accuracy`. Combining full data + optimization takes longer.

**Results:** By default the pipeline optimizes for AUC-ROC (~0.73–0.75 on validation). Validation **accuracy** with the default setup is in the same ballpark. When using `--optimize-for-accuracy`, we get about **67–68% accuracy** on a 25K subset and about **82% accuracy** on the full dataset. The dataset is imbalanced (~18% churn), so a majority-class baseline is already ~82%; higher accuracy would require near-perfect prediction of churners, which the features don’t support. AUC and F1 remain the better metrics for how well the model discriminates churn vs retention.

**Plots written to `results/`:**  
`churn_correlation.png`, `churn_rate_by_category.png`, `feature_importance.png`, `churn_predictors_vs_retention_indicators.png`, `roc_curves.png`, `confusion_matrix.png`.

**Reproducibility:** Random seed 42 for splits and Optuna. Default is the 10K subset; use `--use-full-dataset` for the numbers you’d report.

**Credits:** Dataset from Kaggle (title above). Code in `src/` is original for this project; no copied snippets beyond normal library use. For grading, repo access is granted to lorenzo.destefani@gmail.com per course instructions.
