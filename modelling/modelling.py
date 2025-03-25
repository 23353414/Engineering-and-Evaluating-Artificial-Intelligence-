from model.randomforest import RandomForest
from model.chainedforest import ChainedRandomForest
from model.hierarchicalforest import HierarchicalRandomForest
from Config import Config


def model_predict(data, df, group_name):
    """
    Train models and generate predictions based on the selected approach
    
    Args:
        data: Data object containing training and testing data
        df: DataFrame containing the original data
        group_name: Name of the current data group being processed
    """
    if Config.MODEL_APPROACH == "chained":
        # Multi-label classification approach (Design Choice 1)
        print("\n===== MULTI-LABEL CLASSIFICATION (CHAINED) =====")
        chained_model = ChainedRandomForest("ChainedRandomForest", data.get_embeddings(), data.get_type())
        
        # Train on multi-label combinations
        chained_model.train_multi_label(data)
        
        # Predict for test data
        chained_model.predict_multi_label(data=data)
        
        # Print results
        chained_model.print_multi_label_results(data)
    
    elif Config.MODEL_APPROACH == "hierarchical":
        # Hierarchical classification approach (Design Choice 2)
        print("\n===== MULTI-LABEL CLASSIFICATION (HIERARCHICAL) =====")
        hierarchical_model = HierarchicalRandomForest("HierarchicalRandomForest", data.get_embeddings(), data.get_type())
        
        # Train on hierarchical data
        hierarchical_model.train_multi_label(data)
        
        # Predict for test data
        hierarchical_model.predict_multi_label(data=data)
        
        # Print results
        hierarchical_model.print_multi_label_results(data)
    
    else:
        # Original single-label approach
        print("\n===== SINGLE-LABEL CLASSIFICATION =====")
        print("RandomForest")
        single_model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
        single_model.train(data)
        single_model.predict(data.X_test)
        single_model.print_results(data)


def model_evaluate(model, data):
    """
    Evaluate model performance based on model type
    
    Args:
        model: The trained model to evaluate
        data: Data object containing testing data
    """
    if Config.ENABLE_MULTI_LABEL and Config.MODEL_APPROACH == "chained" and isinstance(model, ChainedRandomForest):
        model.print_multi_label_results(data)
    elif Config.ENABLE_MULTI_LABEL and Config.MODEL_APPROACH == "hierarchical" and isinstance(model, HierarchicalRandomForest):
        model.print_multi_label_results(data)
    else:
        model.print_results(data)