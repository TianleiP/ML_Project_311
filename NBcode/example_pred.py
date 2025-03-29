"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict, Counter





def clean_complexity(value):
    try:
        value = float(value)
        if 1 <= value <= 5:
            return value
    except:
        pass
    return np.nan

def clean_ingredients(text):
    if not isinstance(text, str):
        return np.nan
    numbers = re.findall(r'\d+', text)
    if len(numbers) == 1:
        return int(numbers[0])
    elif len(numbers) == 2:
        return (int(numbers[0]) + int(numbers[1])) / 2
    else:
        return np.nan

def extract_settings(text):
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
    
    text_lower = text.lower()
    if 'week day lunch' in text_lower:
        settings['weekday_lunch'] = 1
    if 'week day dinner' in text_lower:
        settings['weekday_dinner'] = 1
    if 'weekend lunch' in text_lower:
        settings['weekend_lunch'] = 1
    if 'weekend dinner' in text_lower:
        settings['weekend_dinner'] = 1
    if 'at a party' in text_lower:
        settings['party'] = 1
    if 'late night snack' in text_lower:
        settings['late_night'] = 1
    
    return settings

def clean_price(text):
    if not isinstance(text, str):
        return np.nan
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if len(numbers) == 1:
        return float(numbers[0])
    elif len(numbers) == 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    else:
        return np.nan

def clean_movie(text):
    if not isinstance(text, str):
        return np.nan
    words = text.split()
    if len(words) <= 8:
        return text
    return np.nan

def normalize_drink(drink):
    if not isinstance(drink, str):
        return 'other'
    drink = drink.lower().strip()
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
    for main_term, variants in drink_groups.items():
        if any(variant == drink or variant in drink.split() for variant in variants):
            return main_term
    return 'other'

def extract_who_reminds(text):
    reminds = {'parents': 0, 'siblings': 0, 'friends': 0, 'teachers': 0, 'strangers': 0}
    if not isinstance(text, str):
        return reminds
    text_lower = text.lower()
    if 'parents' in text_lower:
        reminds['parents'] = 1
    if 'siblings' in text_lower:
        reminds['siblings'] = 1
    if 'friends' in text_lower:
        reminds['friends'] = 1
    if 'teachers' in text_lower:
        reminds['teachers'] = 1
    if 'strangers' in text_lower:
        reminds['strangers'] = 1
    return reminds

def clean_hot_sauce_numeric(text):
    if not isinstance(text, str):
        return np.nan
    text_lower = text.lower()
    if 'none' in text_lower:
        return 1
    elif 'little' in text_lower or 'mild' in text_lower:
        return 2
    elif 'moderate' in text_lower or 'medium' in text_lower:
        return 3
    elif 'lot' in text_lower or 'hot' in text_lower:
        return 4
    elif 'with my hot sauce' in text_lower:
        return 5
    else:
        return np.nan

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_survey_data_for_inference(
    input_file,
    existing_vocabulary,
    existing_group_medians=None
):
    """
    Process data in "test/inference" mode: 
      - We do not save intermediate CSVs
      - We do not build a new vocabulary or group medians
      - We do bag-of-words using the existing_vocabulary
      - We fill numeric missing values using existing_group_medians 
        or fallback with median of that column
    
    Returns a DataFrame with final features.
    """
    print(f"Reading test data from {input_file}...")
    df = pd.read_csv(input_file, quotechar='"', escapechar='\\', on_bad_lines='skip', low_memory=False)
    print(f"Data shape: {df.shape}")

    # Create a new DataFrame
    processed = pd.DataFrame()

    # If 'id' is present, keep it
    if 'id' in df.columns:
        processed['id'] = df['id']
    else:
        processed['id'] = range(len(df))

    # If 'Label' is present, we won't use it for inference, but let's keep it for reference
    processed['Label'] = df['Label'] if 'Label' in df.columns else np.nan

    # Q1
    col_q1 = 'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'
    if col_q1 in df.columns:
        processed['complexity'] = df[col_q1].apply(clean_complexity)
    else:
        processed['complexity'] = np.nan

    # Q2
    col_q2 = 'Q2: How many ingredients would you expect this food item to contain?'
    if col_q2 in df.columns:
        processed['ingredients'] = df[col_q2].apply(clean_ingredients)
    else:
        processed['ingredients'] = np.nan

    # Q3: settings
    col_q3 = 'Q3: In what setting would you expect this food to be served? Please check all that apply'
    if col_q3 in df.columns:
        settings_series = df[col_q3].apply(extract_settings)
        settings_df = pd.DataFrame.from_records(settings_series.tolist())
    else:
        settings_df = pd.DataFrame(columns=['weekday_lunch','weekday_dinner','weekend_lunch','weekend_dinner','party','late_night'])
        for c in settings_df.columns:
            settings_df[c] = 0
    processed = pd.concat([processed, settings_df], axis=1)

    # Q4: price
    col_q4 = 'Q4: How much would you expect to pay for one serving of this food item?'
    if col_q4 in df.columns:
        processed['price'] = df[col_q4].apply(clean_price)
    else:
        processed['price'] = np.nan

    # Q5: movie
    col_q5 = 'Q5: What movie do you think of when thinking of this food item?'
    if col_q5 in df.columns:
        processed['movie'] = df[col_q5].apply(clean_movie)
    else:
        processed['movie'] = np.nan
    processed['movie'] = processed['movie'].fillna('other')

    # Q6: drink
    col_q6 = 'Q6: What drink would you pair with this food item?'
    if col_q6 in df.columns:
        df['normalized_drink'] = df[col_q6].apply(normalize_drink)
        top_drinks = [d for d,_ in Counter(df['normalized_drink'].dropna()).most_common(4)]
        for d in top_drinks:
            processed[f'drink_{d}'] = (df['normalized_drink'] == d).astype(int)
        processed['drink_other'] = (~df['normalized_drink'].isin(top_drinks)).astype(int)
    else:
        processed['drink_other'] = 1  # fallback everything is "other"

    # Q7: who reminds
    col_q7 = 'Q7: When you think about this food item, who does it remind you of?'
    if col_q7 in df.columns:
        reminds_series = df[col_q7].apply(extract_who_reminds)
        reminds_df = pd.DataFrame.from_records(reminds_series.tolist())
    else:
        reminds_df = pd.DataFrame(columns=['parents','siblings','friends','teachers','strangers'])
        for c in reminds_df.columns:
            reminds_df[c] = 0
    processed = pd.concat([processed, reminds_df], axis=1)

    # Q8: hot sauce
    col_q8 = 'Q8: How much hot sauce would you add to this food item?'
    if col_q8 in df.columns:
        processed['hot_sauce'] = df[col_q8].apply(clean_hot_sauce_numeric)
    else:
        processed['hot_sauce'] = np.nan

    # Fill numeric columns using group_medians OR fallback to global median
    numeric_cols = ['ingredients','price','hot_sauce']
    if existing_group_medians is None:
        existing_group_medians = {}  # fallback if truly missing

    def fill_numeric(row):
        label = row['Label']
        for c in numeric_cols:
            if pd.isna(row[c]):
                if label in existing_group_medians:
                    row[c] = existing_group_medians[label][c]
                else:
                    # fallback: global median of that column
                    row[c] = processed[c].median()
        return row

    processed = processed.apply(fill_numeric, axis=1)

    # Bag-of-words for Q5: movie, using existing vocabulary
    cleaned_text = processed['movie'].apply(clean_text)
    tokenized = cleaned_text.apply(lambda x: x.split())

    bow_data = []
    for tokens in tokenized:
        word_counts = defaultdict(int)
        for w in tokens:
            word_counts[w] += 1
        row_features = {}
        for w, c in word_counts.items():
            if w in existing_vocabulary:
                row_features[f"movie_{w}"] = c
        bow_data.append(row_features)

    bow_df = pd.DataFrame(bow_data).fillna(0)
    bow_df.reset_index(drop=True, inplace=True)
    processed.reset_index(drop=True, inplace=True)

    for col in bow_df.columns:
        processed[col] = bow_df[col].values

    return processed


###############################################
def load_nb_params():
    """
    Reads naive_bayes_priors.csv and naive_bayes_likelihoods.csv
    from the current folder and returns two dictionaries:
    
        priors_dict = {
            'Pizza': <float log_prior>,
            'Shawarma': <float log_prior>,
            'Sushi': <float log_prior>,
        }
    
        likelihoods_dict = {
            'Pizza':    {'movie_word1': <float>, 'movie_word2': <float>, ... },
            'Shawarma': { ... },
            'Sushi':    { ... }
        }
    """
    priors_df = pd.read_csv("naive_bayes_priors.csv")  
    # Expecting columns: [label, log_prior]
    priors_dict = {}
    for _, row in priors_df.iterrows():
        label = str(row["label"])
        priors_dict[label] = float(row["log_prior"])
    
    # 2) Read likelihoods
    # Expecting columns: [label, feature_name, log_likelihood]
    likelihoods_df = pd.read_csv("naive_bayes_likelihoods.csv")
    likelihoods_dict = {}
    for _, row in likelihoods_df.iterrows():
        label = str(row["label"])
        feature = str(row["feature_name"])
        ll_value = float(row["log_likelihood"])
        if label not in likelihoods_dict:
            likelihoods_dict[label] = {}
        likelihoods_dict[label][feature] = ll_value
    
    return priors_dict, likelihoods_dict


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    priors_dict, likelihoods_dict = load_nb_params()
    label_map = {
        "Pizza": 0,
        "Shawarma": 1,
        "Sushi": 2
    }

    # We'll accumulate log scores in a dict
    log_scores = {}

    # For each label, start with log_prior
    for label in priors_dict.keys():
        log_sum = priors_dict[label]

        # For each feature in x
        # (We assume x[feature] is a numeric count or continuous value.)
        for feature_name, feature_value in x.items():
            # If the feature_value is 0 or NaN, it won't affect the sum
            # If there's no known log_likelihood for this feature under the label, skip it
            if feature_value != 0 and not pd.isna(feature_value):
                if feature_name in likelihoods_dict[label]:
                    log_sum += feature_value * likelihoods_dict[label][feature_name]

        log_scores[label] = log_sum

    # Pick the label with the highest log score
    best_label = max(log_scores, key=log_scores.get)

    # Convert the best_label string to its numeric code
    return best_label


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    
    vocab_df = pd.read_csv("vocabulary.csv")  # same folder
    vocab_list = vocab_df["word"].tolist()
    
    test_df = process_survey_data_for_inference(
        input_file=filename,
        existing_vocabulary=vocab_list
    )

    predictions = []
    # Iterate through each row as (index, Series)
    for _, test_example in test_df.iterrows():
        # Obtain a prediction for this test example (single row)
        pred = predict(test_example)
        predictions.append(pred)
    
    return predictions


##################################################################################

# If your labels are EXACTLY these strings:
label_map = {
    "Pizza": 0,
    "Shawarma": 1,
    "Sushi": 2
}

def compute_accuracy(csv_file):
    """
    1. Read the CSV file (which contains 'Label').
    2. Convert each row's Label to numeric code using 'label_map'.
    3. Call 'predict_all(filename)' to get predictions (list of ints).
    4. Compare predicted vs. actual numeric codes to compute accuracy.
    """
    # 1) Read the CSV
    df = pd.read_csv(csv_file)
    # 2) Convert ground truth labels ("Pizza"/"Shawarma"/"Sushi") to numeric
    df["label_code"] = df["Label"].map(label_map)

    # 3) Get predictions from your existing function
    #    e.g. [0, 2, 1, ...]
    preds = predict_all(csv_file)
    print(preds)
    # 4) Compare
    # ensure both are the same length
    if len(preds) != len(df):
        raise ValueError("Number of predictions != number of rows in the CSV.")

    # Convert preds to a pandas Series so we can compare easily
    preds_series = pd.Series(preds, index=df.index)

    # Check correctness (True if match, False if mismatch)
    correct = (df["label_code"] == preds_series)

    # Accuracy = (# correct) / (total)
    accuracy = correct.mean()

    return accuracy

# ------------------------------
# EXAMPLE USAGE:
# ------------------------------
if __name__ == "__main__":
    # 1) Define the path
    csv_file = r"C:\Users\kevin\Desktop\311\ML_Project_311\data\cleaned_data_combined_modified.csv"
    
    # 2) Compute accuracy
    acc = compute_accuracy(csv_file)
    
    # 3) Print result
    print(f"Accuracy on '{csv_file}': {acc:.4f}")