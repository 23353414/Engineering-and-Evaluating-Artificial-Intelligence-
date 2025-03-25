class Config:
    """
    Configuration settings for the multi-label email classification system
    
    This class centralizes all configuration parameters to make the system easily configurable.
    Modify these settings to:
    1. Choose which approach to use (chained, hierarchical, or both)
    2. Enable/disable multi-label classification
    3. Define the label combinations to use
    4. Adjust model hyperparameters
    """
    
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
    
    # Data groups to process
    DATA_GROUPS = ['AppGallery & Games', 'In-App Purchase']
    
    # Data file path
    DATA_PATH = 'out.csv'
    
    # Multi-label classification settings
    ENABLE_MULTI_LABEL = True
    
    # Model approach options:
    # - "chained" for Design Choice 1 only
    # - "hierarchical" for Design Choice 2 only
    # - "both" to run both approaches sequentially and compare results
    MODEL_APPROACH = "both"
    
    # Label combinations for multi-label classification
    # Each entry defines a level of prediction:
    # - ['y2'] for Type 2 only
    # - ['y2', 'y3'] for Type 2 + Type 3
    # - ['y2', 'y3', 'y4'] for Type 2 + Type 3 + Type 4
    LABEL_COMBINATIONS = [
        ['y2'],                 # Type 2 only
        ['y2', 'y3'],           # Type 2 + Type 3
        ['y2', 'y3', 'y4']      # Type 2 + Type 3 + Type 4
    ]
    
    # Separator for combined labels (used in the chained approach)
    # This string separates the different label components in combined labels
    # Example: "Problem/Fault_+_Payment_+_Subscription cancellation"
    SEPARATOR = '_+_'           
    
    # Random seed for reproducibility
    # Using a fixed seed ensures consistent results across runs
    RANDOM_SEED = 0
    
    # Minimum samples required for a class to be considered
    # Classes with fewer than this number of samples will be ignored
    # This helps prevent overfitting to extremely rare classes
    MIN_SAMPLES_PER_CLASS = 3
    
    # Model hyperparameters
    N_ESTIMATORS = 1000         # Number of trees in the random forest
    CLASS_WEIGHT = 'balanced_subsample'  # Weight classes inversely to their frequency