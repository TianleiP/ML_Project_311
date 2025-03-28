import numpy as np
import pandas as pd
import joblib
import json
import os
import pickle

def extract_tree_structure(tree):
    """
    Extract the structure of a decision tree.
    Returns a dictionary with the nodes, features, thresholds and values.
    """
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    values = tree.value
    
    tree_structure = {
        'n_nodes': n_nodes,
        'children_left': children_left.tolist(),
        'children_right': children_right.tolist(),
        'feature': feature.tolist(),
        'threshold': threshold.tolist(),
        'values': [v.tolist() for v in values]
    }
    
    return tree_structure

def extract_random_forest_params(model_path, output_path):
    """
    Extract parameters from a trained Random Forest model and save them
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    rf_model = joblib.load(model_path)
    
    # Extract parameters
    n_estimators = len(rf_model.estimators_)
    classes = rf_model.classes_.tolist()
    
    print(f"Model has {n_estimators} trees and {len(classes)} classes")
    
    # Extract structure of each tree
    trees = []
    for i, tree in enumerate(rf_model.estimators_):
        print(f"Extracting tree {i+1}/{n_estimators}...")
        tree_structure = extract_tree_structure(tree.tree_)
        trees.append(tree_structure)
    
    # Create a parameters dictionary
    params = {
        'n_estimators': n_estimators,
        'classes': classes,
        'trees': trees
    }
    
    # Save parameters to file
    with open(output_path, 'w') as f:
        json.dump(params, f)
    
    print(f"Parameters saved to {output_path}")
    return params

def predict_tree(tree_params, features):
    """
    Make a prediction with a single decision tree
    """
    # Get tree parameters
    children_left = tree_params['children_left']
    children_right = tree_params['children_right']
    feature_indices = tree_params['feature']
    thresholds = tree_params['threshold']
    values = tree_params['values']
    
    # Start from the root node (index 0)
    node_id = 0
    
    # Traverse the tree
    while True:
        # Check if we've reached a leaf node
        if children_left[node_id] == -1:
            # Leaf node - return the predicted class
            value = values[node_id]
            return np.argmax(value[0])
        
        # Get the feature to check
        feature_idx = feature_indices[node_id]
        
        # If feature is -2, it's a leaf
        if feature_idx == -2:
            value = values[node_id]
            return np.argmax(value[0])
        
        # Get feature value
        feature_value = features[feature_idx]
        
        # Check the threshold
        if feature_value <= thresholds[node_id]:
            # Go left
            node_id = children_left[node_id]
        else:
            # Go right
            node_id = children_right[node_id]

def predict_random_forest(params_path, input_data):
    """
    Predict using a Random Forest without scikit-learn
    """
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    n_estimators = params['n_estimators']
    classes = params['classes']
    trees = params['trees']
    
    # Make predictions with each tree
    all_predictions = []
    for i, tree_params in enumerate(trees):
        # Convert input data to numpy array
        if isinstance(input_data, pd.DataFrame):
            features = input_data.values
        else:
            features = input_data
            
        # Make predictions for each sample
        tree_predictions = []
        for sample in features:
            prediction = predict_tree(tree_params, sample)
            tree_predictions.append(prediction)
        
        all_predictions.append(tree_predictions)
    
    # Convert to numpy array
    all_predictions = np.array(all_predictions)
    
    # Majority vote
    final_predictions = []
    for i in range(all_predictions.shape[1]):
        # Get predictions for this sample from all trees
        tree_votes = all_predictions[:, i]
        
        # Count votes
        unique_values, counts = np.unique(tree_votes, return_counts=True)
        
        # Get majority vote
        majority_vote = unique_values[np.argmax(counts)]
        final_predictions.append(classes[majority_vote])
    
    return np.array(final_predictions)

def save_prediction_function():
    """
    Save the prediction function to be used in pred.py
    """
    with open('rf_prediction_function.py', 'w') as f:
        f.write("""
import numpy as np
import json

def predict_tree(tree_params, features):
    \"\"\"
    Make a prediction with a single decision tree
    \"\"\"
    # Get tree parameters
    children_left = tree_params['children_left']
    children_right = tree_params['children_right']
    feature_indices = tree_params['feature']
    thresholds = tree_params['threshold']
    values = tree_params['values']
    
    # Start from the root node (index 0)
    node_id = 0
    
    # Traverse the tree
    while True:
        # Check if we've reached a leaf node
        if children_left[node_id] == -1:
            # Leaf node - return the predicted class
            value = values[node_id]
            return np.argmax(value[0])
        
        # Get the feature to check
        feature_idx = feature_indices[node_id]
        
        # If feature is -2, it's a leaf
        if feature_idx == -2:
            value = values[node_id]
            return np.argmax(value[0])
        
        # Get feature value
        feature_value = features[feature_idx]
        
        # Check the threshold
        if feature_value <= thresholds[node_id]:
            # Go left
            node_id = children_left[node_id]
        else:
            # Go right
            node_id = children_right[node_id]

def random_forest_predict(X, params_path='rf_params.json'):
    \"\"\"
    Predict using a Random Forest without scikit-learn
    \"\"\"
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    n_estimators = params['n_estimators']
    classes = params['classes']
    trees = params['trees']
    
    # Make predictions with each tree
    all_predictions = []
    for i, tree_params in enumerate(trees):
        # Convert input data to numpy array
        if hasattr(X, 'values'):
            features = X.values
        else:
            features = X
            
        # Make predictions for each sample
        tree_predictions = []
        for sample in features:
            prediction = predict_tree(tree_params, sample)
            tree_predictions.append(prediction)
        
        all_predictions.append(tree_predictions)
    
    # Convert to numpy array
    all_predictions = np.array(all_predictions)
    
    # Majority vote
    final_predictions = []
    for i in range(all_predictions.shape[1]):
        # Get predictions for this sample from all trees
        tree_votes = all_predictions[:, i]
        
        # Count votes
        unique_values, counts = np.unique(tree_votes, return_counts=True)
        
        # Get majority vote
        majority_vote = unique_values[np.argmax(counts)]
        final_predictions.append(int(majority_vote))
    
    return np.array(final_predictions)
""")
    print("Prediction function saved to rf_prediction_function.py")

if __name__ == "__main__":
    # Check if model exists
    model_path = 'random_forest_model.joblib'
    output_path = 'rf_params.json'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        print("Please run random_forest_model.py first to train and save the model.")
        exit(1)
    
    # Extract parameters
    params = extract_random_forest_params(model_path, output_path)
    
    # Save prediction function
    save_prediction_function()
    
    # Test prediction
    try:
        # Try to load a sample of the original dataset
        df = pd.read_csv("data/final_combined_data/final_data_with_bow.csv")
        df = df.drop(columns=['id', 'Label']).head(5)
        
        print("\nTesting prediction function on sample data...")
        predictions = predict_random_forest(output_path, df)
        print(f"Sample predictions: {predictions}")
        
        print("\nExtraction and testing completed successfully.")
        print(f"To use the extracted model in pred.py, import the functions from rf_prediction_function.py")
        print(f"and load the parameters from {output_path}")
    except Exception as e:
        print(f"Error testing prediction: {str(e)}")
        print("Parameters have been extracted successfully, but testing failed.") 