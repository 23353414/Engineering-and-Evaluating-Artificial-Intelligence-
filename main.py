from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
import time
import pandas as pd
from sys import exit
from Config import Config
from model.chainedforest import ChainedRandomForest
from model.hierarchicalforest import HierarchicalRandomForest

# Set random seed for reproducibility
random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


def load_data():
    """Load the input data from CSV files"""
    df = get_input_data()
    return df


def preprocess_data(df):
    """Preprocess the input data"""
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    # Translate data to english (commented out to speed up processing)
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    """Generate embeddings for the input data"""
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create a data object for modeling"""
    return Data(X, df)


def perform_modelling(data, df, group_name):
    """Perform modelling on the given data"""
    if data.X_train is None or data.y_train is None:
        print(f"No training data available for {group_name}")
        return
        
    # Predict
    model_predict(data, df, group_name)
    
    # Evaluate
    model_evaluate(data, group_name)
    
    print("="*25 + "\n")


def print_config_info():
    """Prints the current configuration settings"""
    print("\n===== CONFIGURATION =====")
    print(f"Multi-label classification enabled: {Config.ENABLE_MULTI_LABEL}")
    
    approach_desc = {
        "chained": "Chained approach only (Design Choice 1)",
        "hierarchical": "Hierarchical approach only (Design Choice 2)",
        "both": "Running both approaches sequentially"
    }
    print(f"Model approach: {approach_desc.get(Config.MODEL_APPROACH, 'Unknown')}")
    
    print("Label combinations:")
    for combo in Config.LABEL_COMBINATIONS:
        # Handle both string and list types
        if isinstance(combo, list):
            print(f"- {' + '.join(combo)}")
        else:
            print(f"- {combo.replace('_+_', ' + ')}")
    print("=========================\n\n")


def print_summary_comparison(chained_results, hierarchical_results):
    """
    Prints a formatted comparison of accuracy results between approaches
    
    Args:
        chained_results: Dictionary with accuracy results from chained approach
        hierarchical_results: Dictionary with accuracy results from hierarchical approach
    """
    print("\n" + "="*70)
    print(" "*20 + "ACCURACY COMPARISON BETWEEN APPROACHES")
    print("="*70)
    
    # Determine the label combinations to display
    all_combos = set(list(chained_results.keys()) + list(hierarchical_results.keys()))
    ordered_combos = ["y2", "y2_+_y3", "y2_+_y3_+_y4"]
    
    # Print header
    print(f"{'Label Combination':<30} {'Chained Approach':<20} {'Hierarchical Approach':<20} {'Difference':<15}")
    print("-"*85)
    
    # Print results for each combination
    for combo in ordered_combos:
        if combo in all_combos:
            chained_acc = chained_results.get(combo, "N/A")
            hier_acc = hierarchical_results.get(combo, "N/A")
            
            if chained_acc != "N/A" and hier_acc != "N/A":
                diff = hier_acc - chained_acc
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "N/A"
            
            # Format the combination name for display
            combo_display = combo.replace('_+_', ' + ').replace('y2', 'Type 2').replace('y3', 'Type 3').replace('y4', 'Type 4')
            
            # Format the values
            chained_str = f"{chained_acc:.4f}" if chained_acc != "N/A" else "N/A"
            hier_str = f"{hier_acc:.4f}" if hier_acc != "N/A" else "N/A"
            
            print(f"{combo_display:<30} {chained_str:<20} {hier_str:<20} {diff_str:<15}")
    
    print("="*85)
    print("Positive difference indicates hierarchical approach performed better")
    print("="*85)


def main():
    """Main function to orchestrate the workflow"""
    start_time = time.time()
    print_config_info()
    
    all_chained_results = {}
    all_hierarchical_results = {}
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Convert string columns to unicode to handle special characters
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Group data by Type 1 (y1)
    grouped_df = df.groupby(Config.GROUPED)
    
    for group_name, group_df in grouped_df:
        print(f"\n\n***** Processing group: {group_name} *****")
        
        # Get embeddings and create data object
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        
        # Skip if not enough data
        if data.X_train is None or data.y_train is None:
            print(f"Not enough data for group {group_name}. Skipping...")
            continue
        
        # DESIGN CHOICE 1: Chained approach
        if Config.MODEL_APPROACH in ["chained", "both"]:
            print("\n" + "="*50)
            print("RUNNING CHAINED APPROACH (DESIGN CHOICE 1)")
            print("="*50 + "\n")
            
            chained_model = ChainedRandomForest("ChainedRandomForest", None, None)
            
            print("===== MULTI-LABEL CLASSIFICATION (CHAINED) =====")
            chained_model.train_multi_label(data)
            chained_model.predict_multi_label(data=data)
            chained_model.print_multi_label_results(data)
            
            # Save results for comparison
            all_chained_results.update(chained_model.accuracy)
        
        # DESIGN CHOICE 2: Hierarchical approach
        if Config.MODEL_APPROACH in ["hierarchical", "both"]:
            print("\n" + "="*50)
            print("RUNNING HIERARCHICAL APPROACH (DESIGN CHOICE 2)")
            print("="*50 + "\n")
            
            hierarchical_model = HierarchicalRandomForest("HierarchicalRandomForest", None, None)
            
            print("===== MULTI-LABEL CLASSIFICATION (HIERARCHICAL) =====")
            hierarchical_model.train_multi_label(data)
            hierarchical_model.predict_multi_label(data=data)
            hierarchical_model.print_multi_label_results(data)
            
            # Save results for comparison
            all_hierarchical_results.update(hierarchical_model.accuracy)
        
        print("")
    
    # If both approaches were run, print comparison
    if Config.MODEL_APPROACH == "both" and Config.ENABLE_MULTI_LABEL:
        print_summary_comparison(all_chained_results, all_hierarchical_results)
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")


if __name__ == '__main__':
    main()

