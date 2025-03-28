import numpy as np
import pandas as pd
import json
import sys
import os
import re
from collections import Counter, defaultdict

# ===== DATA PROCESSING FUNCTIONS =====

def clean_complexity(value):
    """Q1: Leave it as it is (1-5 scale)."""
    try:
        value = float(value)
        if 1 <= value <= 5:
            return value
    except:
        pass
    return np.nan

def clean_ingredients(text):
    """
    Q2: Extract integers:
    - Keep if one integer
    - Average if two integers
    - Null if more than two integers or no integers
    """
    if not isinstance(text, str):
        return np.nan
    
    # Extract all numbers
    numbers = re.findall(r'\d+', text)
    
    if len(numbers) == 1:
        return int(numbers[0])
    elif len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    else:
        return np.nan

def extract_settings(text):
    """Q3: Process into vector of 6*1"""
    settings = {
        'weekday_lunch': 0,
        'weekday_dinner': 0,
        'weekend_lunch': 0,
        'weekend_dinner': 0,
        'party': 0,
        'late_night': 0
    }
    
    if not isinstance(text, str):
        return settings
    
    text = text.lower()
    
    if 'week day lunch' in text:
        settings['weekday_lunch'] = 1
    if 'week day dinner' in text:
        settings['weekday_dinner'] = 1
    if 'weekend lunch' in text:
        settings['weekend_lunch'] = 1
    if 'weekend dinner' in text:
        settings['weekend_dinner'] = 1
    if 'at a party' in text:
        settings['party'] = 1
    if 'late night snack' in text:
        settings['late_night'] = 1
    
    return settings

def clean_price(text):
    """
    Q4: Extract integers:
    - Keep if one integer
    - Average if two integers
    - Null if more than two integers or no integers
    """
    if not isinstance(text, str):
        return np.nan
    
    # Extract all numbers
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    
    if len(numbers) == 1:
        return float(numbers[0])
    elif len(numbers) == 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    else:
        return np.nan

def clean_text(text):
    """Clean text for bag-of-words processing."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep letters, numbers and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def normalize_drink(drink):
    """Q6: Group similar drinks into categories."""
    if not isinstance(drink, str):
        return 'other'
    
    drink = drink.lower().strip()
    
    # Define groups of similar drinks
    drink_groups = {
        'coca cola': ['coke', 'coca cola', 'diet coke', 'cola', 'coke zero', 'pepsi', 'diet pepsi'],
        'tea': ['tea', 'green tea', 'iced tea', 'ice tea', 'bubble tea', 'milk tea'],
        'water': ['water', 'sparkling water', 'mineral water', 'soda water'],
        'beer': ['beer', 'craft beer', 'cold beer'],
        'juice': ['juice', 'orange juice', 'apple juice', 'fruit juice', 'lemon juice'],
        'soda': ['soda', 'pop', 'soft drink', 'carbonated drink', 'sprite', 'fanta', '7up', 'mountain dew'],
        'wine': ['wine', 'red wine', 'white wine', 'rose wine'],
        'coffee': ['coffee', 'iced coffee', 'espresso', 'latte'],
        'milk': ['milk', 'chocolate milk', 'dairy']
    }
    
    # Check each group for matches
    for main_term, variants in drink_groups.items():
        if any(variant == drink or variant in drink.split() for variant in variants):
            return main_term
    
    return 'other'

def extract_who_reminds(text):
    """Q7: Process into vector of 5*1"""
    reminds = {
        'parents': 0,
        'siblings': 0,
        'friends': 0,
        'teachers': 0,
        'strangers': 0
    }
    
    if not isinstance(text, str):
        return reminds
    
    text = text.lower()
    
    if 'parents' in text:
        reminds['parents'] = 1
    if 'siblings' in text:
        reminds['siblings'] = 1
    if 'friends' in text:
        reminds['friends'] = 1
    if 'teachers' in text:
        reminds['teachers'] = 1
    if 'strangers' in text:
        reminds['strangers'] = 1
    
    return reminds

def clean_hot_sauce_numeric(text):
    """
    Q8: Convert to numeric scale 1-5:
    1 = None
    2 = A little (mild)
    3 = A moderate amount (medium)
    4 = A lot (hot)
    5 = I will have some of this food item with my hot sauce
    """
    if not isinstance(text, str):
        return np.nan
    
    text = text.lower()
    
    if 'none' in text:
        return 1
    elif 'little' in text or 'mild' in text:
        return 2
    elif 'moderate' in text or 'medium' in text:
        return 3
    elif 'lot' in text or 'hot' in text:
        return 4
    elif 'with my hot sauce' in text:
        return 5
    else:
        return np.nan

# ===== FEATURE MANAGEMENT FUNCTIONS =====

def load_feature_names():
    """
    Load feature names from feature_names.csv or feature_names_with_index.csv
    Returns a list of feature names excluding 'id' and 'Label'
    """
    feature_files = ['feature_names.csv']
    
    for file in feature_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                if 'Column_Name' in df.columns:
                    feature_names = df['Column_Name'].tolist()
                    
                    # Remove 'id' and 'Label' if they exist
                    if 'id' in feature_names:
                        feature_names.remove('id')
                    if 'Label' in feature_names:
                        feature_names.remove('Label')
                    
                    print(f"Loaded {len(feature_names)} feature names from {file}")
                    return feature_names
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    # If feature_names.csv doesn't exist, use a hardcoded list of features
    print("Warning: No feature names file found. Using hardcoded feature list.")
    return ["complexity", "ingredients", "weekday_lunch", "weekday_dinner", "weekend_lunch", 
            "weekend_dinner", "party", "late_night", "price", "drink_coca cola", "drink_water", 
            "drink_other", "drink_soda", "parents", "siblings", "friends", "teachers", 
            "strangers", "hot_sauce"]

def get_movie_features():
    """
    Extract movie features from the feature names list
    Returns a list of movie feature names and words
    """
    feature_names = load_feature_names()
    movie_features = [feat for feat in feature_names if feat.startswith('movie_')]
    print(f"Found {len(movie_features)} movie features")
    
    # Extract words from feature names
    movie_words = [feat.replace('movie_', '') for feat in movie_features]
    
    return movie_features, movie_words

def process_movie_text(movie_text, movie_words):
    """
    Process movie text to match the fixed movie features
    Returns a dictionary with movie feature values (0 or 1)
    """
    # Clean the movie text
    cleaned_text = clean_text(movie_text)
    words = cleaned_text.split() if cleaned_text else []
    
    # Initialize all movie features to 0
    movie_features = {f"movie_{word}": 0 for word in movie_words}
    
    # Set features to 1 for words that appear in the text
    for word in words:
        if word in movie_words:
            movie_features[f"movie_{word}"] = 1
    
    return movie_features

# ===== DATA PROCESSING MAIN FUNCTION =====

def process_data(input_file):
    """
    Process raw data to produce features matching exactly the feature_names.csv structure
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, quotechar='"', escapechar='\\', 
                     on_bad_lines='skip', low_memory=False)
    
    print(f"Original data shape: {df.shape}")
    
    # Load feature names to ensure exact matching
    feature_names = load_feature_names()
    
    # Get movie features and words
    movie_features, movie_words = get_movie_features()
    
    # Initialize result DataFrame
    result_df = pd.DataFrame()
    
    # Keep the id column if it exists
    if 'id' in df.columns:
        result_df['id'] = df['id']
    
    # Keep the Label column if it exists
    if 'Label' in df.columns:
        result_df['Label'] = df['Label']
    
    # Process each feature systematically
    
    # 1. Complexity
    print("Processing Q1: Complexity...")
    result_df['complexity'] = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].apply(clean_complexity)
    
    # 2. Ingredients
    print("Processing Q2: Ingredients...")
    result_df['ingredients'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_ingredients)
    
    # 3. Settings
    print("Processing Q3: Settings...")
    settings = df['Q3: In what setting would you expect this food to be served? Please check all that apply'].apply(extract_settings)
    settings_df = pd.DataFrame.from_records(settings.tolist())
    for col in settings_df.columns:
        result_df[col] = settings_df[col]
    
    # 4. Price
    print("Processing Q4: Price...")
    result_df['price'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(clean_price)
    
    # 5. Drinks
    print("Processing Q6: Drinks...")
    drink_col = 'Q6: What drink would you pair with this food item?'
    normalized_drinks = df[drink_col].apply(normalize_drink)
    
    # Use fixed drink features from feature_names
    drink_features = [f for f in feature_names if f.startswith('drink_')]
    for feature in drink_features:
        drink_name = feature.replace('drink_', '')
        result_df[feature] = (normalized_drinks == drink_name).astype(int)
    
    # 6. Who Reminds
    print("Processing Q7: Who reminds...")
    reminds = df['Q7: When you think about this food item, who does it remind you of?'].apply(extract_who_reminds)
    reminds_df = pd.DataFrame.from_records(reminds.tolist())
    for col in reminds_df.columns:
        result_df[col] = reminds_df[col]
    
    # 7. Hot Sauce
    print("Processing Q8: Hot sauce...")
    result_df['hot_sauce'] = df['Q8: How much hot sauce would you add to this food item?'].apply(clean_hot_sauce_numeric)
    
    # 8. Movies - using fixed feature set
    print("Processing Q5: Movie with fixed features...")
    movie_col = 'Q5: What movie do you think of when thinking of this food item?'
    
    # Initialize all movie features to 0
    for feature in movie_features:
        result_df[feature] = 0
    
    # Process each movie
    for i, movie_text in enumerate(df[movie_col]):
        if pd.notna(movie_text) and isinstance(movie_text, str):
            # Clean text and limit to 8 words
            words = movie_text.split()
            if len(words) <= 8:
                movie_text_features = process_movie_text(movie_text, movie_words)
                for feature, value in movie_text_features.items():
                    if feature in result_df.columns:
                        result_df.at[i, feature] = value
    
    # Fill missing values
    print("Filling missing values...")
    
    # For numerical features, use median
    for col in ['complexity', 'ingredients', 'price', 'hot_sauce']:
        if col in result_df.columns and result_df[col].isna().any():
            median_val = result_df[col].median()
            result_df[col] = result_df[col].fillna(median_val)
            print(f"  - Filled {col} NULLs with median: {median_val:.2f}")
    
    # For categorical features, already filled with 0s
    
    # Ensure all expected features are present
    missing_features = set(feature_names) - set(result_df.columns)
    if missing_features:
        print(f"Warning: {len(missing_features)} features from feature_names.csv are missing")
        for feature in missing_features:
            print(f"  - Adding missing feature: {feature}")
            result_df[feature] = 0
    
    # Ensure column order matches feature_names exactly
    result_cols = []
    if 'id' in result_df.columns:
        result_cols.append('id')
    if 'Label' in result_df.columns:
        result_cols.append('Label')
    
    # Add remaining features in the exact order from feature_names
    for feature in feature_names:
        if feature in result_df.columns:
            result_cols.append(feature)
    
    # Reorder columns
    result_df = result_df[result_cols]
    
    print(f"Final processed data shape: {result_df.shape}")
    return result_df

# ===== RANDOM FOREST PREDICTION =====

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
        
        # Ensure feature_idx is within the features array bounds
        if feature_idx >= len(features):
            # Default to class 0 if feature index is out of bounds
            return 0
            
        # Get feature value
        feature_value = features[feature_idx]
        
        # Check the threshold
        if feature_value <= thresholds[node_id]:
            # Go left
            node_id = children_left[node_id]
        else:
            # Go right
            node_id = children_right[node_id]

def predict(X):
    """
    Make predictions using Random Forest
    """
    # Load parameters
    params_path = 'rf_params.json'
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file {params_path} not found. Run random_forest_extractor.py first.")
    
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
    
    return final_predictions

def predict_all(filename):
    """
    Process data and make predictions
    """
    # Process the raw data
    print("Processing raw data...")
    processed_data = process_data(filename)
    
    # Save processed data for reference
    processed_file = "processed_data_aligned.csv"
    processed_data.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")
    
    # Prepare data for prediction
    pred_data = processed_data.copy()
    
    # Remove id and Label if present
    if 'id' in pred_data.columns:
        ids = pred_data['id'].values
        pred_data = pred_data.drop(columns=['id'])
    else:
        ids = np.arange(1, len(pred_data) + 1)
    
    if 'Label' in pred_data.columns:
        pred_data = pred_data.drop(columns=['Label'])
    
    # Make predictions
    print("Making predictions...")
    numeric_predictions = predict(pred_data)
    
    # Map to food categories
    food_categories = ["Pizza", "Shawarma", "Sushi"]
    string_predictions = [food_categories[pred] for pred in numeric_predictions]
    
    print(f"Made predictions for {len(string_predictions)} samples")
    
    # Store IDs for use in main function
    global prediction_ids
    prediction_ids = ids
    
    return string_predictions

# ===== MAIN FUNCTION =====

if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "cleaned_data_combined.csv"  # Default path
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        
        # Try looking in data directory
        data_path = os.path.join("data", os.path.basename(file_path))
        if os.path.exists(data_path):
            print(f"Using file from data directory: {data_path}")
            file_path = data_path
        else:
            print("Available CSV files:")
            for file in os.listdir("."):
                if file.endswith(".csv"):
                    print(f"  - {file}")
            if os.path.exists("data"):
                print("Files in data directory:")
                for file in os.listdir("data"):
                    if file.endswith(".csv"):
                        print(f"  - data/{file}")
            sys.exit(1)
    
    # Get predictions
    predictions = predict_all(file_path)
    
    # Get IDs from global variable
    ids = prediction_ids
    
    # Print sample predictions
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:10]):
        print(f"Sample {ids[i]}: {pred}")
    
    # Save predictions to CSV
    output_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })
    output_file = 'predictions_aligned.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Count predictions by category
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    print("\nPrediction distribution:")
    for category, count in pred_counts.items():
        percent = (count / len(predictions)) * 100
        print(f"  {category}: {count} ({percent:.1f}%)") 