"""
Robust Data Preprocessing Pipeline with strict zero-leakage guarantee.
Fits imputers, scalers, and encoders exclusively on training partitions.
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Transforms raw tabular data into numerical matrices ready for machine learning.
    Ensures zero data leakage by isolating fit() to the training fold.
    """
    
    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
        # Transformers
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        
        # State
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.datetime_features: List[str] = []
        self.id_features: List[str] = []
        self.encoded_feature_names: List[str] = []
        self.is_fitted: bool = False

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        drop_cols: Optional[List[str]] = None,
        stratify: bool = False
    ) -> Dict[str, Any]:
        """
        Splits data and applies zero-leakage transformation.
        """
        df_clean = df.copy()
        
        # 1. Column discovery
        if drop_cols is None:
            drop_cols = []
            
        # Discover ID-like and datetime features
        self.id_features = []
        self.datetime_features = []
        for col in df_clean.columns:
            if col == target_col or col in drop_cols:
                continue
            col_lower = str(col).lower()
            if any(k in col_lower for k in ["id", "uuid", "key", "identifier", "code"]):
                if df_clean[col].nunique() / len(df_clean) > 0.8:
                    self.id_features.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                self.datetime_features.append(col)
            elif df_clean[col].dtype == object and any(k in col_lower for k in ["date", "time", "timestamp"]):
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                    if df_clean[col].notnull().mean() > 0.7:
                        self.datetime_features.append(col)
                except Exception:
                    pass

        # Decompose datetime columns into numerical components
        for dt_col in self.datetime_features:
            dt_series = pd.to_datetime(df_clean[dt_col], errors="coerce")
            df_clean[f"{dt_col}_month"] = dt_series.dt.month.fillna(1).astype(int)
            df_clean[f"{dt_col}_day"] = dt_series.dt.day.fillna(1).astype(int)
            df_clean[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
            df_clean[f"{dt_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        # Exclude IDs, target, and raw datetime objects from feature matrix
        exclude_cols = set(drop_cols + self.id_features + self.datetime_features)
        if target_col:
            exclude_cols.add(target_col)
            
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
        
        self.numeric_features = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c])
        ]
        self.categorical_features = [
            c for c in feature_cols if c not in self.numeric_features
        ]
        
        X = df_clean[feature_cols]
        y = df_clean[target_col] if target_col else None
        
        if y is not None:
            # Handle missing values in target
            valid_idx = y.notnull()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
            strat = y if stratify and y.nunique() <= 10 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=strat
            )
            
            # Fit only on train
            X_train_proc = self._fit_transform_features(X_train)
            X_test_proc = self._transform_features(X_test)
            
            return {
                "X_train": X_train_proc,
                "X_test": X_test_proc,
                "y_train": y_train,
                "y_test": y_test,
                "X_train_raw": X_train,
                "X_test_raw": X_test,
                "feature_names": self.encoded_feature_names,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features
            }
        else:
            # Unsupervised (Clustering / Anomaly Detection)
            X_proc = self._fit_transform_features(X)
            return {
                "X": X_proc,
                "X_raw": X,
                "feature_names": self.encoded_feature_names,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features
            }

    def _fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        matrices = []
        encoded_names = []
        
        # Numeric pipeline
        if self.numeric_features:
            num_data = X[self.numeric_features].copy()
            num_imputed = self.num_imputer.fit_transform(num_data)
            num_scaled = self.scaler.fit_transform(num_imputed)
            matrices.append(num_scaled)
            encoded_names.extend(self.numeric_features)
            
        # Categorical pipeline
        if self.categorical_features:
            cat_data = X[self.categorical_features].astype(str)
            cat_imputed = self.cat_imputer.fit_transform(cat_data)
            cat_encoded = self.encoder.fit_transform(cat_imputed)
            matrices.append(cat_encoded)
            encoded_cat_names = self.encoder.get_feature_names_out(self.categorical_features).tolist()
            encoded_names.extend(encoded_cat_names)
            
        self.encoded_feature_names = encoded_names
        self.is_fitted = True
        
        if not matrices:
            raise ValueError("No valid features found for modeling.")
            
        return np.hstack(matrices)

    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
            
        matrices = []
        if self.numeric_features:
            num_data = X[self.numeric_features].copy()
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.scaler.transform(num_imputed)
            matrices.append(num_scaled)
            
        if self.categorical_features:
            cat_data = X[self.categorical_features].astype(str)
            cat_imputed = self.cat_imputer.transform(cat_data)
            cat_encoded = self.encoder.transform(cat_imputed)
            matrices.append(cat_encoded)
            
        return np.hstack(matrices)

    def transform_single(self, input_df: pd.DataFrame) -> np.ndarray:
        """Transforms new input records for what-if simulations."""
        # Add required engineered datetime cols if needed
        df_copy = input_df.copy()
        for dt_col in self.datetime_features:
            if dt_col in df_copy.columns:
                dt_series = pd.to_datetime(df_copy[dt_col], errors="coerce")
                df_copy[f"{dt_col}_month"] = dt_series.dt.month.fillna(1).astype(int)
                df_copy[f"{dt_col}_day"] = dt_series.dt.day.fillna(1).astype(int)
                df_copy[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(int)
                df_copy[f"{dt_col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                
        # Fill any missing columns from training
        for col in self.numeric_features:
            if col not in df_copy.columns:
                df_copy[col] = 0.0
        for col in self.categorical_features:
            if col not in df_copy.columns:
                df_copy[col] = "Unknown"
                
        return self._transform_features(df_copy)
