import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from utils import load_data, get_feature_types

# Columns added in create_feature_engineering — not part of raw API / CSV input.
ENGINEERED_FEATURE_NAMES = frozenset(
    {
        "TotalChargesPerMonth",
        "MonthlyChargesRatio",
        "ViewingHoursPerMonth",
        "ViewingIntensity",
        "SupportIntensity",
        "ContentEngagement",
        "LowRating",
        "HighSupportUsage",
    }
)


class ChurnDataPreprocessor:
    """Load, impute, encode, split. Label encoding, median/mode imputation, StandardScaler, 80/20."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.feature_types = None
        self.imputer_numerical = None
        self.imputer_categorical = None
        self.support_tickets_q75 = None

    def load_and_prepare_data(self, data_path="../customer-churn-dataset/"):
        train_data, test_data, _ = load_data(data_path)
        train_data['is_train'] = True
        test_data['is_train'] = False
        return pd.concat([train_data, test_data], ignore_index=True)

    def analyze_features(self, data):
        self.feature_types = get_feature_types(data)
        return self.feature_types

    def _numerical_columns(self):
        return list(self.feature_types['numerical']) if self.feature_types else []

    def _categorical_columns(self):
        if not self.feature_types:
            return []
        return self.feature_types['categorical'] + self.feature_types['binary']

    def handle_missing_values(self, data, fit_imputers=True):
        """Fit or apply median/mode imputation. Always persist imputers when fit_imputers=True."""
        numerical_features = self._numerical_columns()
        categorical_features = self._categorical_columns()

        if fit_imputers:
            if numerical_features:
                self.imputer_numerical = SimpleImputer(strategy='median')
                data[numerical_features] = self.imputer_numerical.fit_transform(data[numerical_features])
            else:
                self.imputer_numerical = None

            if categorical_features:
                self.imputer_categorical = SimpleImputer(strategy='most_frequent')
                data[categorical_features] = self.imputer_categorical.fit_transform(data[categorical_features])
            else:
                self.imputer_categorical = None
        else:
            if self.imputer_numerical is not None and numerical_features:
                data[numerical_features] = self.imputer_numerical.transform(data[numerical_features])
            if self.imputer_categorical is not None and categorical_features:
                data[categorical_features] = self.imputer_categorical.transform(data[categorical_features])

        return data

    def encode_categorical_features(self, data, fit_encoders=True):
        categorical_features = self._categorical_columns()
        for feature in categorical_features:
            if fit_encoders:
                label_encoder = LabelEncoder()
                data[feature] = label_encoder.fit_transform(data[feature].astype(str))
                self.label_encoders[feature] = label_encoder
            else:
                if feature in self.label_encoders:
                    label_encoder = self.label_encoders[feature]
                    data[feature] = data[feature].astype(str)
                    has_unseen_values = ~data[feature].isin(label_encoder.classes_)
                    if has_unseen_values.any():
                        data.loc[has_unseen_values, feature] = label_encoder.classes_[0]
                    data[feature] = label_encoder.transform(data[feature])
        return data

    def create_feature_engineering(self, data, fit_stats=True):
        if fit_stats and 'SupportTicketsPerMonth' in data.columns:
            self.support_tickets_q75 = float(data['SupportTicketsPerMonth'].quantile(0.75))

        if 'MonthlyCharges' in data.columns and 'AccountAge' in data.columns:
            data['TotalChargesPerMonth'] = data['TotalCharges'] / (data['AccountAge'] + 1)
            data['MonthlyChargesRatio'] = data['MonthlyCharges'] / data['TotalChargesPerMonth']

        if 'ViewingHoursPerWeek' in data.columns and 'AccountAge' in data.columns:
            data['ViewingHoursPerMonth'] = data['ViewingHoursPerWeek'] * 4.33
            data['ViewingIntensity'] = data['ViewingHoursPerMonth'] / (data['AccountAge'] + 1)

        if 'SupportTicketsPerMonth' in data.columns and 'AccountAge' in data.columns:
            data['SupportIntensity'] = data['SupportTicketsPerMonth'] / (data['AccountAge'] + 1)

        if 'ContentDownloadsPerMonth' in data.columns and 'ViewingHoursPerWeek' in data.columns:
            data['ContentEngagement'] = data['ContentDownloadsPerMonth'] / (data['ViewingHoursPerWeek'] + 1)

        if 'UserRating' in data.columns:
            data['LowRating'] = (data['UserRating'] < 3).astype(int)

        if 'SupportTicketsPerMonth' in data.columns:
            q75 = self.support_tickets_q75
            if q75 is None:
                q75 = float(data['SupportTicketsPerMonth'].quantile(0.75))
            data['HighSupportUsage'] = (data['SupportTicketsPerMonth'] > q75).astype(int)

        return data

    def raw_input_column_names(self):
        """Feature columns expected in ``transform_raw`` input before engineering (sorted)."""
        if not self.feature_names:
            raise RuntimeError("feature_names unset; run select_features after training or load artifacts.")
        return sorted(c for c in self.feature_names if c not in ENGINEERED_FEATURE_NAMES)

    def select_features(self, data):
        exclude_columns = ['CustomerID', 'is_train', 'Churn']
        self.feature_names = [col for col in data.columns if col not in exclude_columns]
        return self.feature_names

    def scale_features(self, X_train, X_val, X_test, scaling_method='standard'):
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError("scaling_method must be 'standard'")

        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_names, index=X_train.index)
        X_val = pd.DataFrame(self.scaler.transform(X_val), columns=self.feature_names, index=X_val.index)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_names, index=X_test.index)

        return X_train, X_val, X_test

    def split_data(self, data, feature_columns):
        """Stratified 80/20 train/val from is_train=True; test from is_train=False."""
        X = data[feature_columns]
        train_mask = data['is_train'] == True
        test_mask = data['is_train'] == False

        X_train_original = X[train_mask]
        X_test_original = X[test_mask]

        if 'Churn' in data.columns:
            y = data['Churn']
            y_train_original = y[train_mask]
            y_test_original = y[test_mask] if test_mask.sum() > 0 else pd.Series(dtype=float)
        else:
            y_train_original = pd.Series(dtype=float)
            y_test_original = pd.Series(dtype=float)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_original, y_train_original,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_train_original
        )

        return X_train, X_val, X_test_original, y_train, y_val, y_test_original

    def transform_raw(self, raw):
        """
        Apply the same preprocessing as training (impute → encode → engineer → scale).
        Expects a fitted preprocessor (after training or load_preprocessing_artifacts).

        Parameters
        ----------
        raw : dict or pandas.DataFrame
            Raw feature row(s) with columns as in the CSV (before encoding). Churn optional.
            is_train is added as False if missing.

        Returns
        -------
        pandas.DataFrame
            Scaled feature matrix aligned to self.feature_names.
        """
        if self.feature_names is None or self.scaler is None:
            raise RuntimeError("Preprocessor is not fitted: call training pipeline or load_preprocessing_artifacts first.")
        if self.feature_types is None:
            raise RuntimeError("feature_types is missing; analyze_features or load required.")

        if isinstance(raw, dict):
            df = pd.DataFrame([raw])
        else:
            df = raw.copy()

        if 'is_train' not in df.columns:
            df['is_train'] = False

        df = self.handle_missing_values(df, fit_imputers=False)
        df = self.encode_categorical_features(df, fit_encoders=False)
        df = self.create_feature_engineering(df, fit_stats=False)

        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns after preprocessing: {sorted(missing)}")

        X = df[self.feature_names]
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=df.index,
        )

    def save_preprocessing_artifacts(self, save_dir="../results/"):
        import joblib
        save_path = Path(save_dir)
        if not save_path.is_absolute():
            save_path = (Path(__file__).parent / save_path).resolve()
        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.label_encoders, save_path / "label_encoders.pkl")
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        joblib.dump(self.feature_names, save_path / "feature_names.pkl")
        joblib.dump(self.feature_types, save_path / "feature_types.pkl")
        joblib.dump(self.imputer_numerical, save_path / "imputer_numerical.pkl")
        joblib.dump(self.imputer_categorical, save_path / "imputer_categorical.pkl")
        joblib.dump({"support_tickets_q75": self.support_tickets_q75}, save_path / "feature_eng_stats.pkl")

    def load_preprocessing_artifacts(self, save_dir="../results/"):
        import joblib
        load_path = Path(save_dir)
        if not load_path.is_absolute():
            load_path = (Path(__file__).parent / load_path).resolve()

        self.label_encoders = joblib.load(load_path / "label_encoders.pkl")
        self.scaler = joblib.load(load_path / "scaler.pkl")
        self.feature_names = joblib.load(load_path / "feature_names.pkl")
        self.feature_types = joblib.load(load_path / "feature_types.pkl")
        self.imputer_numerical = joblib.load(load_path / "imputer_numerical.pkl")
        self.imputer_categorical = joblib.load(load_path / "imputer_categorical.pkl")
        stats = joblib.load(load_path / "feature_eng_stats.pkl")
        self.support_tickets_q75 = stats.get("support_tickets_q75") if isinstance(stats, dict) else None
