import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    """
    Data encapsulation class that prepares and stores data for model training and evaluation
    
    This class serves multiple purposes:
    1. Encapsulates data to provide consistent access across different models
    2. Handles data preprocessing (filtering classes with too few samples)
    3. Creates and maintains train/test splits consistently
    4. Provides specialized data structures for both multi-label approaches:
       - Chained approach (Design Choice 1): Combined labels like "Type2_+_Type3"
       - Hierarchical approach (Design Choice 2): Filtered datasets for each level
    
    By encapsulating all data handling in this class, we ensure that:
    - Models have a consistent interface for accessing data
    - Data preparation logic is separated from model implementation
    - Train/test splits are maintained consistently across approaches
    """
    
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        """
        Initialize the Data object
        
        Args:
            X: Feature matrix (embeddings)
            df: Dataframe containing labels and other metadata
        """
        # Original single-label approach
        y = df.y.to_numpy()
        y_series = pd.Series(y)

        # Filter out classes with too few samples
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value)<1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        # Keep only samples from classes with enough instances
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        # Calculate test size to maintain the desired proportion after filtering
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        
        # Create train/test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good
        )
        
        # Store original data
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X
        self.df = df
        
        # Multi-label classification approach (Design Choice 1: Chained)
        if Config.ENABLE_MULTI_LABEL:
            # Initialize dictionaries to store multi-label data
            self.multi_labels = {}           # Combined labels for all samples
            self.multi_train_labels = {}     # Combined labels for training samples
            self.multi_test_labels = {}      # Combined labels for testing samples
            self.multi_classes = {}          # Unique classes for each combination
            self.multi_train_indices = {}    # Indices of training samples for each combination
            self.multi_test_indices = {}     # Indices of testing samples for each combination
            
            # Get train and test indices from the original split
            # This ensures consistency across all models
            train_indices = np.arange(len(X_good))[~np.isin(np.arange(len(X_good)), np.arange(len(self.X_test)))]
            test_indices = np.arange(len(X_good))[np.isin(np.arange(len(X_good)), np.arange(len(self.X_test)))]
            
            # Create combined labels for all combinations specified in the config
            for combo in Config.LABEL_COMBINATIONS:
                combo_key = Config.SEPARATOR.join(combo)
                print(f"Processing combination: {combo_key}")
                
                # Check for missing values in the combination columns
                valid_rows = df[combo].notna().all(axis=1)
                df_valid = df[valid_rows].copy()
                
                if df_valid.empty:
                    print(f"No valid data for combination {combo_key}")
                    continue
                
                # Create combined label strings using the separator
                # For example: "Problem/Fault_+_Payment_+_Subscription cancellation"
                df_valid['combined_label'] = df_valid[combo].apply(
                    lambda x: Config.SEPARATOR.join(x.values.astype(str)), axis=1
                )
                
                # Filter for classes with enough samples
                label_counts = df_valid['combined_label'].value_counts()
                good_classes = label_counts[label_counts >= 3].index
                
                if len(good_classes) < 1:
                    print(f"Not enough samples for combination {combo_key}: Skipping...")
                    continue
                
                # Get indices of samples with good classes
                good_indices = df_valid[df_valid['combined_label'].isin(good_classes)].index
                
                # Store the combined labels
                self.multi_labels[combo_key] = df_valid.loc[good_indices, 'combined_label'].values
                self.multi_classes[combo_key] = good_classes
                
                # Use the same train/test split as the single-label approach
                # This ensures consistent samples across all models
                X_indices = np.array(range(len(X)))
                train_mask = np.isin(X_indices, train_indices)
                test_mask = np.isin(X_indices, test_indices)
                
                # Get multi-label data with matching indices
                train_label_indices = np.intersect1d(good_indices, df.index[train_mask])
                test_label_indices = np.intersect1d(good_indices, df.index[test_mask])
                
                if len(train_label_indices) < 3 or len(test_label_indices) < 1:
                    print(f"Not enough train/test samples for {combo_key}: Skipping...")
                    continue
                
                # Store train/test data and indices
                self.multi_train_indices[combo_key] = train_label_indices
                self.multi_test_indices[combo_key] = test_label_indices
                
                # Extract labels for training and testing
                self.multi_train_labels[combo_key] = df_valid.loc[train_label_indices, 'combined_label'].values
                self.multi_test_labels[combo_key] = df_valid.loc[test_label_indices, 'combined_label'].values
                
                print(f"Created {len(self.multi_train_labels[combo_key])} training and {len(self.multi_test_labels[combo_key])} testing samples for {combo_key}")

            # Hierarchical classification approach (Design Choice 2)
            # For each level, store mappings and filtered data
            self.hierarchical_labels = {}           # Labels for each hierarchical subset
            self.hierarchical_train_labels = {}     # Training labels for each subset
            self.hierarchical_test_labels = {}      # Testing labels for each subset
            self.hierarchical_classes = {}          # Unique classes for each subset
            self.hierarchical_train_indices = {}    # Training indices for each subset
            self.hierarchical_test_indices = {}     # Testing indices for each subset
            
            # Process Type 2 (top level)
            top_level = 'y2'
            if top_level in df.columns:
                # Check for missing values
                valid_rows = df[top_level].notna()
                df_valid = df[valid_rows].copy()
                
                if not df_valid.empty:
                    # Filter for classes with enough samples
                    label_counts = df_valid[top_level].value_counts()
                    good_classes = label_counts[label_counts >= 3].index
                    
                    if len(good_classes) >= 1:
                        # Get indices of samples with good classes
                        good_indices = df_valid[df_valid[top_level].isin(good_classes)].index
                        
                        # Store the labels and classes
                        self.hierarchical_labels[top_level] = df_valid.loc[good_indices, top_level].values
                        self.hierarchical_classes[top_level] = good_classes
                        
                        # Get train/test split for hierarchical data
                        X_indices = np.array(range(len(X)))
                        train_mask = np.isin(X_indices, train_indices)
                        test_mask = np.isin(X_indices, test_indices)
                        
                        # Get hierarchical data with matching indices
                        train_label_indices = np.intersect1d(good_indices, df.index[train_mask])
                        test_label_indices = np.intersect1d(good_indices, df.index[test_mask])
                        
                        if len(train_label_indices) >= 3 and len(test_label_indices) >= 1:
                            # Store train/test data and indices
                            self.hierarchical_train_indices[top_level] = train_label_indices
                            self.hierarchical_test_indices[top_level] = test_label_indices
                            
                            # Extract labels for training and testing
                            self.hierarchical_train_labels[top_level] = df_valid.loc[train_label_indices, top_level].values
                            self.hierarchical_test_labels[top_level] = df_valid.loc[test_label_indices, top_level].values
                            
                            print(f"Created {len(self.hierarchical_train_labels[top_level])} training and {len(self.hierarchical_test_labels[top_level])} testing samples for hierarchical {top_level}")
                            
                            # For each Type 2 class, create filtered datasets for Type 3
                            for class_value in good_classes:
                                # Create subset key (e.g., "y2=Problem/Fault")
                                subset_key = f"{top_level}={class_value}"
                                
                                # Filter data for this Type 2 class
                                type2_indices = df_valid[df_valid[top_level] == class_value].index
                                
                                # Keep only rows with valid Type 3 values
                                valid_type3_rows = df.loc[type2_indices, 'y3'].notna()
                                type3_indices = type2_indices[valid_type3_rows]
                                
                                if len(type3_indices) >= 3:
                                    # Store the labels for this subset
                                    self.hierarchical_labels[subset_key] = df.loc[type3_indices, 'y3'].values
                                    
                                    # Get unique Type 3 classes for this subset
                                    type3_values = df.loc[type3_indices, 'y3'].value_counts()
                                    type3_good_classes = type3_values[type3_values >= 3].index
                                    
                                    if len(type3_good_classes) >= 1:
                                        # Store classes for this subset
                                        self.hierarchical_classes[subset_key] = type3_good_classes
                                        
                                        # Filter for good Type 3 classes
                                        type3_good_indices = type3_indices[df.loc[type3_indices, 'y3'].isin(type3_good_classes)]
                                        
                                        # Get train/test split for this subset
                                        train_subset_indices = np.intersect1d(type3_good_indices, train_label_indices)
                                        test_subset_indices = np.intersect1d(type3_good_indices, test_label_indices)
                                        
                                        if len(train_subset_indices) >= 3 and len(test_subset_indices) >= 1:
                                            # Store train/test data and indices
                                            self.hierarchical_train_indices[subset_key] = train_subset_indices
                                            self.hierarchical_test_indices[subset_key] = test_subset_indices
                                            
                                            # Extract labels for training and testing
                                            self.hierarchical_train_labels[subset_key] = df.loc[train_subset_indices, 'y3'].values
                                            self.hierarchical_test_labels[subset_key] = df.loc[test_subset_indices, 'y3'].values
                                            
                                            print(f"Created {len(train_subset_indices)} training and {len(test_subset_indices)} testing samples for hierarchical {subset_key}")
                                            
                                            # For each Type 3 class in this subset, create filtered datasets for Type 4
                                            for class3_value in type3_good_classes:
                                                # Create subset key for Type 4 (e.g., "y2=Problem/Fault,y3=Payment")
                                                subset3_key = f"{subset_key},y3={class3_value}"
                                                
                                                # Filter data for this Type 3 class
                                                type3_class_indices = type3_good_indices[df.loc[type3_good_indices, 'y3'] == class3_value]
                                                
                                                # Keep only rows with valid Type 4 values
                                                valid_type4_rows = df.loc[type3_class_indices, 'y4'].notna()
                                                type4_indices = type3_class_indices[valid_type4_rows]
                                                
                                                if len(type4_indices) >= 3:
                                                    # Store the labels for this subset
                                                    self.hierarchical_labels[subset3_key] = df.loc[type4_indices, 'y4'].values
                                                    
                                                    # Get unique Type 4 classes for this subset
                                                    type4_values = df.loc[type4_indices, 'y4'].value_counts()
                                                    type4_good_classes = type4_values[type4_values >= 3].index
                                                    
                                                    if len(type4_good_classes) >= 1:
                                                        # Store classes for this subset
                                                        self.hierarchical_classes[subset3_key] = type4_good_classes
                                                        
                                                        # Filter for good Type 4 classes
                                                        type4_good_indices = type4_indices[df.loc[type4_indices, 'y4'].isin(type4_good_classes)]
                                                        
                                                        # Get train/test split for this subset
                                                        train_subset3_indices = np.intersect1d(type4_good_indices, train_label_indices)
                                                        test_subset3_indices = np.intersect1d(type4_good_indices, test_label_indices)
                                                        
                                                        if len(train_subset3_indices) >= 3 and len(test_subset3_indices) >= 1:
                                                            # Store train/test data and indices
                                                            self.hierarchical_train_indices[subset3_key] = train_subset3_indices
                                                            self.hierarchical_test_indices[subset3_key] = test_subset3_indices
                                                            
                                                            # Extract labels for training and testing
                                                            self.hierarchical_train_labels[subset3_key] = df.loc[train_subset3_indices, 'y4'].values
                                                            self.hierarchical_test_labels[subset3_key] = df.loc[test_subset3_indices, 'y4'].values
                                                            
                                                            print(f"Created {len(train_subset3_indices)} training and {len(test_subset3_indices)} testing samples for hierarchical {subset3_key}")


    def get_type(self):
        """Get the target variable"""
        return self.y
        
    def get_X_train(self):
        """Get training features"""
        return self.X_train
        
    def get_X_test(self):
        """Get testing features"""
        return self.X_test
        
    def get_type_y_train(self):
        """Get training labels"""
        return self.y_train
        
    def get_type_y_test(self):
        """Get testing labels"""
        return self.y_test
        
    def get_train_df(self):
        """Get training dataframe"""
        return self.train_df
        
    def get_embeddings(self):
        """Get all feature embeddings"""
        return self.embeddings
        
    def get_type_test_df(self):
        """Get testing dataframe"""
        return self.test_df
        
    def get_X_DL_test(self):
        """Get deep learning test features"""
        return self.X_DL_test
        
    def get_X_DL_train(self):
        """Get deep learning training features"""
        return self.X_DL_train
    
    # Methods for multi-label classification (Design Choice 1: Chained approach)
    def get_multi_label_combinations(self):
        """Returns all available label combinations"""
        if hasattr(self, 'multi_labels'):
            return list(self.multi_labels.keys())
        return []
    
    def get_multi_train_labels(self, combo_key):
        """Get training labels for a specific combination"""
        if hasattr(self, 'multi_train_labels') and combo_key in self.multi_train_labels:
            return self.multi_train_labels[combo_key]
        return None
    
    def get_multi_test_labels(self, combo_key):
        """Get testing labels for a specific combination"""
        if hasattr(self, 'multi_test_labels') and combo_key in self.multi_test_labels:
            return self.multi_test_labels[combo_key]
        return None
    
    def get_multi_classes(self, combo_key):
        """Get classes for a specific combination"""
        if hasattr(self, 'multi_classes') and combo_key in self.multi_classes:
            return self.multi_classes[combo_key]
        return None
        
    def get_multi_train_X(self, combo_key):
        """Get training features for a specific combination"""
        if hasattr(self, 'multi_train_indices') and combo_key in self.multi_train_indices:
            indices = self.multi_train_indices[combo_key]
            return self.embeddings[indices]
        return None
        
    def get_multi_test_X(self, combo_key):
        """Get testing features for a specific combination"""
        if hasattr(self, 'multi_test_indices') and combo_key in self.multi_test_indices:
            indices = self.multi_test_indices[combo_key]
            return self.embeddings[indices]
        return None
    
    def get_multi_test_indices(self, combo_key):
        """Get testing indices for a specific combination"""
        if hasattr(self, 'multi_test_indices') and combo_key in self.multi_test_indices:
            return self.multi_test_indices[combo_key]
        return None
    
    # Methods for hierarchical classification (Design Choice 2)
    def get_hierarchical_label_combinations(self):
        """Returns all available hierarchical label combinations"""
        if hasattr(self, 'hierarchical_labels'):
            return list(self.hierarchical_labels.keys())
        return []
    
    def get_hierarchical_train_labels(self, subset_key):
        """Get training labels for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_train_labels') and subset_key in self.hierarchical_train_labels:
            return self.hierarchical_train_labels[subset_key]
        return None
    
    def get_hierarchical_test_labels(self, subset_key):
        """Get testing labels for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_test_labels') and subset_key in self.hierarchical_test_labels:
            return self.hierarchical_test_labels[subset_key]
        return None
    
    def get_hierarchical_classes(self, subset_key):
        """Get classes for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_classes') and subset_key in self.hierarchical_classes:
            return self.hierarchical_classes[subset_key]
        return None
        
    def get_hierarchical_train_X(self, subset_key):
        """Get training features for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_train_indices') and subset_key in self.hierarchical_train_indices:
            indices = self.hierarchical_train_indices[subset_key]
            return self.embeddings[indices]
        return None
        
    def get_hierarchical_test_X(self, subset_key):
        """Get testing features for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_test_indices') and subset_key in self.hierarchical_test_indices:
            indices = self.hierarchical_test_indices[subset_key]
            return self.embeddings[indices]
        return None
    
    def get_hierarchical_train_indices(self, subset_key):
        """Get training indices for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_train_indices') and subset_key in self.hierarchical_train_indices:
            return self.hierarchical_train_indices[subset_key]
        return None
        
    def get_hierarchical_test_indices(self, subset_key):
        """Get testing indices for a specific hierarchical subset"""
        if hasattr(self, 'hierarchical_test_indices') and subset_key in self.hierarchical_test_indices:
            return self.hierarchical_test_indices[subset_key]
        return None

