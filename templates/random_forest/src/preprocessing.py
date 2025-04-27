from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = SelectKBest(score_func=f_classif)
        self.pca = PCA(n_components=0.95)
        
    def prepare_data(self, data, feature_columns, target_column):
        # Create a copy of the data
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Analyze each column and apply necessary preprocessing
        for column in X.columns:
            X[column] = self._preprocess_column(X[column])
            
        # Handle categorical features if any exist
        if X.select_dtypes(include=['object']).columns.any():
            X = self._encode_features(X)
            
        # Feature selection if more than 5 features
        if X.shape[1] > 5:
            X = self._select_features(X, y)
            
        # Dimensionality reduction if needed (many features)
        if X.shape[1] > 10:
            X = self._reduce_dimensions(X)
            
        # Handle imbalanced target if needed
        X, y = self._balance_classes(X, y)
        
        return X, y
    
    def _preprocess_column(self, column):
        # Check data type
        if pd.api.types.is_numeric_dtype(column):
            # Check for missing values
            if column.isnull().any():
                column = pd.Series(self.imputer.fit_transform(column.values.reshape(-1, 1)).ravel())
            
            # Check for outliers using IQR
            if self._has_outliers(column):
                column = self._handle_outliers_for_column(column)
            
            # Check if scaling is needed (check variance and range)
            if self._needs_scaling(column):
                column = pd.Series(self.scaler.fit_transform(column.values.reshape(-1, 1)).ravel())
                
        return column
    
    def _has_outliers(self, column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((column < (Q1 - 1.5 * IQR)) | (column > (Q3 + 1.5 * IQR)))
        return outlier_condition.any()
    
    def _needs_scaling(self, column):
        # Check if data is already roughly standardized
        mean_close_to_zero = abs(column.mean()) > 0.1
        std_close_to_one = abs(column.std() - 1) > 0.1
        return mean_close_to_zero or std_close_to_one
    
    def _handle_outliers_for_column(self, column, threshold=3):
        mean = column.mean()
        std = column.std()
        
        # Replace outliers with upper/lower bounds
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        return column.clip(lower_bound, upper_bound)
    
    def _encode_features(self, X):
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            # Check cardinality
            unique_count = X[column].nunique()
            
            if unique_count <= 10:  # Use one-hot encoding for low cardinality
                onehot_encoded = pd.get_dummies(X[column], prefix=column)
                X = pd.concat([X.drop(column, axis=1), onehot_encoded], axis=1)
            else:  # Use label encoding for high cardinality
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                X[column] = self.label_encoders[column].fit_transform(X[column].astype(str))
        
        return X
    
    def _select_features(self, X, y, k='all'):
        if isinstance(X, pd.DataFrame):
            original_columns = X.columns
            X = X.to_numpy()
        
        if k == 'all':
            k = min(X.shape[1], 10)  # Limit to top 10 features by default
            
        selected_features = self.feature_selector.fit_transform(X, y)
        selected_indices = self.feature_selector.get_support()
        
        if isinstance(original_columns, pd.Index):
            selected_columns = original_columns[selected_indices]
            return pd.DataFrame(selected_features, columns=selected_columns)
        return pd.DataFrame(selected_features)
    
    def _reduce_dimensions(self, X):
        # Check if PCA is actually needed
        explained_var_ratio = np.var(X, axis=0) / np.var(X, axis=0).sum()
        if (explained_var_ratio < 0.01).any():  # If any feature explains less than 1% of variance
            reduced_features = self.pca.fit_transform(X)
            return pd.DataFrame(reduced_features)
        return X
    
    def _balance_classes(self, X, y):
        # Check if dataset is imbalanced
        class_counts = pd.Series(y).value_counts()
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        
        if max_samples / min_samples > 1.5:  # If imbalanced
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        
        return X, y
    
    def inverse_transform_labels(self, column, encoded_values):
        if column in self.label_encoders:
            return self.label_encoders[column].inverse_transform(encoded_values)
        return encoded_values