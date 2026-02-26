import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from utils import load_data, get_feature_types

class ChurnDataPreprocessor:
    """Load, clean, encode, and split churn data. Label encoding for categories (GBDT-friendly), median/mode imputation, StandardScaler, 80/20 train/val."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.feature_types = None
        
    def load_and_prepare_data(self, data_path="../customer-churn-dataset/"):
        train_data, test_data, _ = load_data(data_path)
        train_data['is_train'] = True
        test_data['is_train'] = False
        return pd.concat([train_data, test_data], ignore_index=True)
    
    def analyze_features(self, data):
        self.feature_types = get_feature_types(data)
        return self.feature_types
    
    def handle_missing_values(self, data):
        if data.isnull().sum().sum() > 0:
            numerical_features = self.feature_types['numerical']
            if numerical_features:
                imputer = SimpleImputer(strategy='median')
                data[numerical_features] = imputer.fit_transform(data[numerical_features])
            
            categorical_features = self.feature_types['categorical'] + self.feature_types['binary']
            if categorical_features:
                imputer = SimpleImputer(strategy='most_frequent')  # mode for categorical
                data[categorical_features] = imputer.fit_transform(data[categorical_features])
        return data
    
    def encode_categorical_features(self, data, fit_encoders=True):
        categorical_features = self.feature_types['categorical'] + self.feature_types['binary']
        for feature in categorical_features:
            if fit_encoders:
                label_encoder = LabelEncoder()
                data[feature] = label_encoder.fit_transform(data[feature].astype(str))
                self.label_encoders[feature] = label_encoder
            else:
                if feature in self.label_encoders:
                    label_encoder = self.label_encoders[feature]
                    data[feature] = data[feature].astype(str)
                    # unseen categories at predict time → map to first class to avoid crash
                    has_unseen_values = ~data[feature].isin(label_encoder.classes_)
                    if has_unseen_values.any():
                        data.loc[has_unseen_values, feature] = label_encoder.classes_[0]
                    data[feature] = label_encoder.transform(data[feature])
        return data
    
    def create_feature_engineering(self, data):
        # derived rates and flags that might help the model
        if 'MonthlyCharges' in data.columns and 'AccountAge' in data.columns:
            data['TotalChargesPerMonth'] = data['TotalCharges'] / (data['AccountAge'] + 1)  # +1 avoids div by 0 for new accounts
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
            data['HighSupportUsage'] = (data['SupportTicketsPerMonth'] > data['SupportTicketsPerMonth'].quantile(0.75)).astype(int)
        
        return data
    
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
        """Train/val from is_train=True (stratified 80/20), test from is_train=False."""
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
    
    def save_preprocessing_artifacts(self, save_dir="../results/"):
        pass

    def load_preprocessing_artifacts(self, save_dir="../results/"):
        import joblib
        self.label_encoders = joblib.load(f"{save_dir}/label_encoders.pkl")
        self.scaler = joblib.load(f"{save_dir}/scaler.pkl")
        self.feature_names = joblib.load(f"{save_dir}/feature_names.pkl")
        self.feature_types = joblib.load(f"{save_dir}/feature_types.pkl")
