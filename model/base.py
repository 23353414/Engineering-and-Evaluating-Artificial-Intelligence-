from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all model implementations
    
    This class defines the interface that all models must implement, ensuring
    consistent access to model functionality regardless of the specific implementation.
    """
    
    def __init__(self) -> None:
        """Initialize the base model"""
        ...


    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using training data
        
        Args:
            data: Data object containing training data
        """
        ...

    @abstractmethod
    def predict(self, X_test) -> None:
        """
        Generate predictions for test data
        
        Args:
            X_test: Test features
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        """Perform any necessary data transformations"""
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
    
    @abstractmethod
    def train_multi_label(self, data, label_combinations) -> None:
        """
        Train the model on multiple label combinations
        
        Args:
            data: Data object containing training data
            label_combinations: List of label combinations to train on
        """
        ...
    
    @abstractmethod
    def predict_multi_label(self, X_test, label_combinations) -> None:
        """
        Generate predictions for multiple label combinations
        
        Args:
            X_test: Test features
            label_combinations: List of label combinations to predict for
        """
        ...
    
    @abstractmethod
    def print_multi_label_results(self, data) -> None:
        """
        Print evaluation results for multiple label combinations
        
        Args:
            data: Data object containing test data and labels
        """
        ...