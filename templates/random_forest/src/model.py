from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

class RandomForestModel:
    def __init__(self, params):
        self.model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        
    def train(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        self.predictions = self.model.predict(self.X_test)
        
    def get_metrics(self):
        return {
            'accuracy': accuracy_score(self.y_test, self.predictions),
            'precision': precision_score(self.y_test, self.predictions, average='weighted'),
            'recall': recall_score(self.y_test, self.predictions, average='weighted')
        }
        
    def get_feature_importance(self):
        return self.model.feature_importances_
        
    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predictions)