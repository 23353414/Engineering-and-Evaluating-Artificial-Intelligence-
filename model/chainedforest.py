import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import *
import random
from Config import Config
import pprint

# Set random seed for reproducibility
random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# Configure pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class ChainedRandomForest(BaseModel):
    """
    Implementation of the chained random forest classifier (Design Choice 1)
    
    This approach uses a single model instance for each label combination:
    - One model for Type 2 classification
    - One model for combined Type 2 + Type 3 classification
    - One model for combined Type 2 + Type 3 + Type 4 classification
    
    Each model predicts a combined label that includes all levels in the chain.
    For example, a model might predict "Problem/Fault_+_Payment_+_Subscription cancellation"
    as a single class.
    
    The key advantages of this approach:
    1. Simpler implementation with fewer models to manage
    2. Consistent training data size across all classification tasks
    3. Direct modeling of label dependencies within a single model
    
    The tradeoffs include:
    1. The number of possible combined classes grows exponentially with each level
    2. Each combined class has fewer examples for training
    3. Cannot leverage the hierarchical structure of the data as directly
    """
    
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialize the ChainedRandomForest model
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
            y: Target labels
        """
        super(ChainedRandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Store model for single label classification
        self.mdl = None
        
        # Dictionaries to store models and predictions for multi-label classification
        self.models = {}         # Stores a model for each label combination
        self.predictions = {}    # Stores predictions for each label combination
        self.accuracy = {}       # Stores accuracy for each label combination

        self.data_transform()

    def train(self, data) -> None:
        """
        Train a single model (for compatibility with existing code)
        
        Args:
            data: Data object containing training data
        """
        model = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS, 
            random_state=Config.RANDOM_SEED,
            class_weight=Config.CLASS_WEIGHT
        )
        self.mdl = model.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        """
        Predict for a single model (for compatibility with existing code)
        
        Args:
            X_test: Test features
        """
        predictions = self.mdl.predict(X_test)
        self.predictions["single"] = predictions

    def print_results(self, data):
        """
        Print results for a single model (for compatibility with existing code)
        
        Args:
            data: Data object containing test data and labels
        """
        print(classification_report(data.y_test, self.predictions["single"], zero_division=0))
        
    def train_multi_label(self, data, label_combinations=None) -> None:
        """
        Train models for multi-label classification using the chained approach
        
        This method:
        1. Creates a separate model for each label combination
        2. Trains each model to predict combined labels directly
        3. For example, one model predicts "Type2", another predicts "Type2_+_Type3"
        
        All models see the same training data, but with different target labels.
        
        Args:
            data: Data object containing training data
            label_combinations: Optional list of label combinations to train on
        """
        if label_combinations is None:
            # Get all possible combinations from data
            label_combinations = Config.LABEL_COMBINATIONS
        
        # Train a model for each label combination
        for combo in label_combinations:
            # Create a key for this combination (e.g., "y2_+_y3_+_y4")
            combo_key = Config.SEPARATOR.join(combo)
            
            # Get training data for this combination
            X_train = data.get_multi_train_X(combo_key)
            y_train = data.get_multi_train_labels(combo_key)
            
            if X_train is None or y_train is None or len(y_train) == 0:
                print(f"No training data for {combo_key}. Skipping...")
                continue
            
            # Create and train the model
            model = RandomForestClassifier(
                n_estimators=Config.N_ESTIMATORS, 
                random_state=Config.RANDOM_SEED,
                class_weight=Config.CLASS_WEIGHT
            )
            
            self.models[combo_key] = model.fit(X_train, y_train)
            print(f"Trained model for {combo_key} with {len(y_train)} samples")

    def predict_multi_label(self, X_test=None, label_combinations=None, data=None) -> None:
        """
        Generate predictions for multiple label combinations using chained models
        
        Each model makes predictions independently for its assigned label combination.
        For example:
        - One model predicts "Problem/Fault" for Type 2
        - Another model predicts "Problem/Fault_+_Payment" for Type 2 + Type 3
        
        This approach directly models each level of specificity as a separate task.
        
        Args:
            X_test: Optional test features
            label_combinations: Optional list of label combinations to predict for
            data: Data object containing test data
        """
        if label_combinations is None:
            # Use all combinations that have models
            label_combinations = list(self.models.keys())
        else:
            # Convert list of lists to list of strings if needed
            label_combinations = [Config.SEPARATOR.join(combo) if isinstance(combo, list) else combo
                                 for combo in label_combinations]
        
        # Generate predictions for each combination
        for combo_key in label_combinations:
            if combo_key not in self.models:
                print(f"No model available for {combo_key}. Skipping...")
                continue
            
            # Get test data for this combination
            X_test = data.get_multi_test_X(combo_key)
            
            if X_test is None or len(X_test) == 0:
                print(f"No test data for {combo_key}. Skipping...")
                continue
            
            # Generate predictions
            predictions = self.models[combo_key].predict(X_test)
            self.predictions[combo_key] = predictions
            print(f"Generated predictions for {combo_key} with {len(predictions)} samples")

    def print_multi_label_results(self, data) -> None:
        """
        Print evaluation results for multiple label combinations
        
        For each label combination, this method:
        1. Retrieves the true test labels and model predictions
        2. Computes and prints a detailed classification report
        3. Calculates and stores the accuracy
        4. Prints a summary of accuracies across all combinations
        
        Args:
            data: Data object containing test data and labels
        """
        print(f"\n===== CHAINED MULTI-LABEL RESULTS FOR {self.model_name} =====")
        
        # Process each combination that has predictions
        for combo_key in self.predictions.keys():
            # Skip any non-label keys
            if not (combo_key == 'y2' or '_+_' in combo_key):
                continue
            
            # Get ground truth labels
            test_y = data.get_multi_test_labels(combo_key)
            
            if test_y is None or len(test_y) == 0:
                print(f"No test labels for {combo_key}. Skipping...")
                continue
            
            # Make sure predictions and labels align
            preds = self.predictions[combo_key]
            
            if len(preds) != len(test_y):
                print(f"Warning: Predictions ({len(preds)}) and test labels ({len(test_y)}) count mismatch for {combo_key}")
                # Use the minimum length to avoid indexing errors
                min_len = min(len(preds), len(test_y))
                preds = preds[:min_len]
                test_y = test_y[:min_len]
            
            # Format the combination name for display
            combo_display = combo_key.replace('_+_', ' + ')
            combo_display = combo_display.replace('y2', 'Type 2').replace('y3', 'Type 3').replace('y4', 'Type 4')
            print(f"\n----- Results for {combo_display} -----")
            
            # Print detailed classification report
            print(classification_report(test_y, preds, zero_division=0))
            
            # Calculate and store accuracy
            accuracy = accuracy_score(test_y, preds)
            self.accuracy[combo_key] = accuracy
            print(f"Accuracy: {accuracy:.4f}")
        
        # Print summary of accuracies
        if self.accuracy:
            print("\n" + "-"*30)
            print("SUMMARY OF ACCURACIES")
            print("-"*30)
            
            # Display accuracies in consistent order
            ordered_keys = ["y2", "y2_+_y3", "y2_+_y3_+_y4"]
            for key in ordered_keys:
                if key in self.accuracy:
                    display_key = key.replace('y2', 'Type 2').replace('y3', 'Type 3').replace('y4', 'Type 4').replace('_+_', ' + ')
                    print(f"{display_key:<20}: {self.accuracy[key]:.4f}")
            
            print("-"*30)

    def data_transform(self) -> None:
        """No transformation needed for this model"""
        pass 