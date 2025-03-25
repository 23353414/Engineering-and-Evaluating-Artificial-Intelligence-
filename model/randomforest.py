import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
from Config import Config

# Set random seed for reproducibility
random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# Configure pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    """
    Implementation of a basic random forest classifier for single-label classification
    """
    
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialize the RandomForest model
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
            y: Target labels
        """
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS, 
            random_state=Config.RANDOM_SEED, 
            class_weight=Config.CLASS_WEIGHT
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Train the model using the provided data
        
        Args:
            data: Data object containing training data
        """
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        """
        Generate predictions for the test data
        
        Args:
            X_test: Test features
        """
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        """
        Print classification results
        
        Args:
            data: Data object containing test data and labels
        """
        print(classification_report(data.y_test, self.predictions, zero_division=0))

    def data_transform(self) -> None:
        """No transformation needed for this model"""
        pass

