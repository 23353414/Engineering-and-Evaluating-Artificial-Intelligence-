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


class HierarchicalRandomForest(BaseModel):
    """
    Implementation of the hierarchical random forest classifier (Design Choice 2)
    
    This approach creates multiple model instances in a hierarchical structure:
    - Top level model for Type 2
    - Separate models for each Type 2 class to predict Type 3
    - Separate models for each Type 2 + Type 3 combination to predict Type 4
    
    The key advantages of this approach:
    1. Models at each level specialize in their specific context
    2. Predictions follow a hierarchical path, matching the inherent structure of the data
    3. Each model only needs to distinguish between classes within its specific context
    
    The tradeoffs include:
    1. More complex implementation with multiple models to manage
    2. Smaller training sets for deeper levels in the hierarchy
    3. Errors can cascade (an error at a higher level affects all predictions below it)
    """
    
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialize the HierarchicalRandomForest model
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
            y: Target labels
        """
        super(HierarchicalRandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Dictionary to store models for different hierarchy levels
        self.models = {}          # Stores trained models for each node in hierarchy
        self.predictions = {}     # Stores predictions for each level and combination
        self.accuracy = {}        # Stores accuracy metrics for each level
        
        # Store class hierarchy relationships for navigation during prediction
        self.hierarchy = {}       # Maps parent classes to their children models
        
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
        Train hierarchical models for multi-label classification
        
        This method creates a tree of models:
        1. First trains a model for Type 2 classification
        2. For each Type 2 class, trains a model for Type 3 classification
        3. For each Type 2 + Type 3 combination, trains a model for Type 4 classification
        
        Each model in the tree is specialized for its specific context.
        
        Args:
            data: Data object containing training data
            label_combinations: Optional list of label combinations to train on
        """
        if label_combinations is None:
            label_combinations = data.get_hierarchical_label_combinations()
        
        # Train Type 2 model first (top level)
        top_level = 'y2'
        top_level_X = data.get_hierarchical_train_X(top_level)
        top_level_y = data.get_hierarchical_train_labels(top_level)
        
        if top_level_X is None or top_level_y is None or len(top_level_y) == 0:
            print(f"No training data for top level {top_level}. Cannot proceed with hierarchical modeling.")
            return
        
        # Create and train top level model
        model = RandomForestClassifier(
            n_estimators=Config.N_ESTIMATORS, 
            random_state=Config.RANDOM_SEED,
            class_weight=Config.CLASS_WEIGHT
        )
        
        self.models[top_level] = model.fit(top_level_X, top_level_y)
        print(f"Trained model for {top_level} with {len(top_level_y)} samples")
        
        # Get unique classes for Type 2
        type2_classes = data.get_hierarchical_classes(top_level)
        
        # For each Type 2 class, train a Type 3 model
        for class_value in type2_classes:
            # Get filtered data for this Type 2 class
            subset_key = f"{top_level}={class_value}"
            subset_X = data.get_hierarchical_train_X(subset_key)
            subset_y = data.get_hierarchical_train_labels(subset_key)
            
            if subset_X is None or subset_y is None or len(subset_y) < Config.MIN_SAMPLES_PER_CLASS:
                print(f"Not enough training data for {subset_key}. Skipping...")
                continue
            
            # Train Type 3 model for this subset
            model = RandomForestClassifier(
                n_estimators=Config.N_ESTIMATORS, 
                random_state=Config.RANDOM_SEED,
                class_weight=Config.CLASS_WEIGHT
            )
            
            self.models[subset_key] = model.fit(subset_X, subset_y)
            print(f"Trained model for {subset_key} with {len(subset_y)} samples")
            
            # Store hierarchy relationship
            if top_level not in self.hierarchy:
                self.hierarchy[top_level] = {}
            self.hierarchy[top_level][class_value] = subset_key
            
            # Get unique classes for Type 3 in this subset
            type3_classes = data.get_hierarchical_classes(subset_key)
            
            # For each Type 3 class, train a Type 4 model
            for class3_value in type3_classes:
                # Get filtered data for this Type 3 class
                subset3_key = f"{subset_key},y3={class3_value}"
                subset3_X = data.get_hierarchical_train_X(subset3_key)
                subset3_y = data.get_hierarchical_train_labels(subset3_key)
                
                if subset3_X is None or subset3_y is None or len(subset3_y) < Config.MIN_SAMPLES_PER_CLASS:
                    print(f"Not enough training data for {subset3_key}. Skipping...")
                    continue
                
                # Train Type 4 model for this subset
                model = RandomForestClassifier(
                    n_estimators=Config.N_ESTIMATORS, 
                    random_state=Config.RANDOM_SEED,
                    class_weight=Config.CLASS_WEIGHT
                )
                
                self.models[subset3_key] = model.fit(subset3_X, subset3_y)
                print(f"Trained model for {subset3_key} with {len(subset3_y)} samples")
                
                # Store hierarchy relationship
                if subset_key not in self.hierarchy:
                    self.hierarchy[subset_key] = {}
                self.hierarchy[subset_key][class3_value] = subset3_key

    def predict_multi_label(self, X_test=None, label_combinations=None, data=None) -> None:
        """
        Generate hierarchical predictions for multiple label combinations
        
        The prediction flow follows the hierarchy structure:
        1. First predict Type 2 using the top-level model
        2. For each instance, find the appropriate Type 3 model based on the Type 2 prediction
        3. Use that model to generate the Type 3 prediction
        4. Similarly, find the Type 4 model based on Type 2 + Type 3 predictions
        
        This cascading approach ensures predictions follow the hierarchical structure.
        
        Args:
            X_test: Optional test features
            label_combinations: Optional list of label combinations to predict for
            data: Data object containing test data
        """
        # First predict Type 2
        top_level = 'y2'
        top_level_X = data.get_hierarchical_test_X(top_level)
        
        if top_level not in self.models or top_level_X is None or len(top_level_X) == 0:
            print(f"No model or test data for top level {top_level}. Cannot generate predictions.")
            return
        
        # Generate Type 2 predictions
        top_preds = self.models[top_level].predict(top_level_X)
        self.predictions[top_level] = top_preds
        print(f"Generated predictions for {top_level} with {len(top_preds)} samples")
        
        # Get the original indices in the DataFrame
        test_indices = data.get_hierarchical_test_indices(top_level)
        
        # Create dictionaries to store predictions for Type 3 and Type 4
        type3_preds = {}
        type4_preds = {}
        
        # Store original df index to prediction mappings
        # This is crucial for properly creating combined predictions later
        df_index_to_pred = {}
        
        # For each Type 2 class, predict using the corresponding Type 3 model
        for class_value in np.unique(top_preds):
            # Get subset key for this Type 2 class
            subset_key = f"{top_level}={class_value}"
            
            if subset_key not in self.models:
                print(f"No model available for {subset_key}. Skipping...")
                continue
            
            # Get indices of test samples predicted as this class
            class_indices = [i for i, pred in enumerate(top_preds) if pred == class_value]
            
            if not class_indices:
                print(f"No test samples predicted as {class_value}. Skipping...")
                continue
            
            # Get test data for these indices
            class_test_indices = [test_indices[i] for i in class_indices]
            class_test_X = data.get_embeddings()[class_test_indices]
            
            # Predict Type 3 for this subset
            type3_subset_preds = self.models[subset_key].predict(class_test_X)
            
            # Store predictions
            for i, pred in zip(class_indices, type3_subset_preds):
                orig_idx = test_indices[i]
                type3_preds[orig_idx] = pred
                df_index_to_pred[orig_idx] = {'y2': top_preds[i], 'y3': pred}
            
            # For each Type 3 class, predict using the corresponding Type 4 model
            for class3_value in np.unique(type3_subset_preds):
                # Get subset key for this Type 3 class
                subset3_key = f"{subset_key},y3={class3_value}"
                
                if subset3_key not in self.models:
                    print(f"No model available for {subset3_key}. Skipping...")
                    continue
                
                # Get indices of test samples predicted as this class
                class3_indices = [class_indices[i] for i, pred in enumerate(type3_subset_preds) if pred == class3_value]
                
                if not class3_indices:
                    print(f"No test samples predicted as {class3_value}. Skipping...")
                    continue
                
                # Get test data for these indices
                class3_test_indices = [test_indices[i] for i in class3_indices]
                class3_test_X = data.get_embeddings()[class3_test_indices]
                
                # Predict Type 4 for this subset
                type4_subset_preds = self.models[subset3_key].predict(class3_test_X)
                
                # Store predictions
                for i, pred in zip(class3_indices, type4_subset_preds):
                    orig_idx = test_indices[i]
                    type4_preds[orig_idx] = pred
                    if orig_idx in df_index_to_pred:
                        df_index_to_pred[orig_idx]['y4'] = pred
        
        # Store predictions
        self.predictions['y3'] = type3_preds
        self.predictions['y4'] = type4_preds
        
        # Create combined label predictions for evaluation
        for combo in Config.LABEL_COMBINATIONS:
            combo_key = Config.SEPARATOR.join(combo)
            
            # Skip if only Type 2 (already handled)
            if combo == ['y2']:
                continue
                
            # Create combined predictions
            combined_preds = []
            combined_indices = []
            
            # For each test instance that has predictions, create combined labels
            for idx, pred_dict in df_index_to_pred.items():
                if combo == ['y2', 'y3']:
                    # Only need Type 2 and Type 3
                    if 'y2' in pred_dict and 'y3' in pred_dict:
                        combined = f"{pred_dict['y2']}{Config.SEPARATOR}{pred_dict['y3']}"
                        combined_preds.append(combined)
                        combined_indices.append(idx)
                elif combo == ['y2', 'y3', 'y4']:
                    # Need all three types
                    if 'y2' in pred_dict and 'y3' in pred_dict and 'y4' in pred_dict:
                        combined = f"{pred_dict['y2']}{Config.SEPARATOR}{pred_dict['y3']}{Config.SEPARATOR}{pred_dict['y4']}"
                        combined_preds.append(combined)
                        combined_indices.append(idx)
            
            if combined_preds:
                self.predictions[combo_key] = combined_preds
                # Store test indices for these predictions to match with test labels later
                self.predictions[f"{combo_key}_indices"] = combined_indices
                print(f"Generated combined predictions for {combo_key} with {len(combined_preds)} samples")

    def print_multi_label_results(self, data) -> None:
        """
        Print evaluation results for hierarchical models
        
        This method:
        1. Creates the appropriate ground truth labels for each prediction level
        2. Compares predictions against these ground truth labels
        3. Calculates and stores accuracy metrics
        4. Prints detailed classification reports and a summary of accuracies
        
        The evaluation handles the special case of combined labels correctly by:
        - Creating combined ground truth labels in the same format as predictions
        - Aligning prediction indices with ground truth indices 
        - Handling potential missing values in the hierarchical structure
        
        Args:
            data: Data object containing test data and labels
        """
        print(f"\n===== HIERARCHICAL MULTI-LABEL RESULTS FOR {self.model_name} =====")
        
        # Print results for Type 2 (top level)
        top_level = 'y2'
        if top_level in self.predictions:
            test_y = data.get_hierarchical_test_labels(top_level)
            test_indices = data.get_hierarchical_test_indices(top_level)
            
            if test_y is not None and len(test_y) > 0:
                print(f"\n----- Results for {top_level.replace('y2', 'Type 2')} -----")
                print(classification_report(test_y, self.predictions[top_level], zero_division=0))
                
                # Calculate and store accuracy
                accuracy = accuracy_score(test_y, self.predictions[top_level])
                self.accuracy[top_level] = accuracy
                print(f"Accuracy: {accuracy:.4f}")
        
        # Print results for combined predictions
        for combo in Config.LABEL_COMBINATIONS:
            combo_key = Config.SEPARATOR.join(combo)
            
            # Skip if only Type 2 (already handled)
            if combo == ['y2']:
                continue
                
            if combo_key in self.predictions:
                # Get the indices of test samples for which we have predictions
                pred_indices = self.predictions.get(f"{combo_key}_indices", [])
                predictions = self.predictions.get(combo_key, [])
                
                if len(pred_indices) == 0 or len(predictions) == 0:
                    continue
                
                # For Type 2 + Type 3 combination
                if combo == ['y2', 'y3']:
                    # Get true combined labels from data
                    true_labels = []
                    
                    # Get raw test data to create combined labels manually
                    test_indices = data.get_hierarchical_test_indices('y2')
                    df = data.df
                    
                    # Create combined ground truth labels
                    for idx in pred_indices:
                        if idx < len(df):
                            # Create combined label string
                            true_label = f"{df.loc[idx, 'y2']}{Config.SEPARATOR}{df.loc[idx, 'y3']}"
                            true_labels.append(true_label)
                
                # For Type 2 + Type 3 + Type 4 combination
                elif combo == ['y2', 'y3', 'y4']:
                    # Get true combined labels from data
                    true_labels = []
                    
                    # Get raw test data to create combined labels manually
                    test_indices = data.get_hierarchical_test_indices('y2')
                    df = data.df
                    
                    # Create combined ground truth labels
                    for idx in pred_indices:
                        if idx < len(df):
                            # Check if all required fields exist
                            if pd.notna(df.loc[idx, 'y2']) and pd.notna(df.loc[idx, 'y3']) and pd.notna(df.loc[idx, 'y4']):
                                # Create combined label string
                                true_label = f"{df.loc[idx, 'y2']}{Config.SEPARATOR}{df.loc[idx, 'y3']}{Config.SEPARATOR}{df.loc[idx, 'y4']}"
                                true_labels.append(true_label)
                
                if len(true_labels) > 0:
                    print(f"Found {len(true_labels)} common records for {combo_key.replace('_+_', ' + ')}")
                    
                    # Make sure predictions match our true labels count
                    valid_predictions = predictions[:len(true_labels)]
                    
                    # Format the combination name for display
                    combo_display = combo_key.replace('_+_', ' + ')
                    print(f"\n----- Results for {combo_display} -----")
                    
                    try:
                        print(classification_report(true_labels, valid_predictions, zero_division=0))
                        
                        # Calculate and store accuracy
                        accuracy = accuracy_score(true_labels, valid_predictions)
                        self.accuracy[combo_key] = accuracy
                        print(f"Accuracy: {accuracy:.4f}")
                    except Exception as e:
                        print(f"Error generating classification report: {e}")
                        print(f"True labels count: {len(true_labels)}")
                        print(f"Predictions count: {len(valid_predictions)}")
                else:
                    print(f"No matching label-prediction pairs for {combo_key}")
        
        # Print summary of accuracies
        if self.accuracy:
            print("\n" + "-"*30)
            print("SUMMARY OF ACCURACIES")
            print("-"*30)
            
            ordered_keys = ["y2", "y2_+_y3", "y2_+_y3_+_y4"]
            for key in ordered_keys:
                if key in self.accuracy:
                    display_key = key.replace('y2', 'Type 2').replace('y3', 'Type 3').replace('y4', 'Type 4').replace('_+_', ' + ')
                    print(f"{display_key:<20}: {self.accuracy[key]:.4f}")
            
            print("-"*30)

    def data_transform(self) -> None:
        """No transformation needed for this model"""
        pass 