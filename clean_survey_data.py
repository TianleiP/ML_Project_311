import pandas as pd
import numpy as np
import re
import os
from collections import Counter

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

def clean_movie(text):
    """Q5: Keep if ≤8 words, else null."""
    if not isinstance(text, str):
        return np.nan
    
    words = text.split()
    if len(words) <= 8:
        return text
    else:
        return np.nan

def normalize_drink(drink):
    """Q6: Group similar drinks into categories."""
    if not isinstance(drink, str):
        return 'other'
    
    drink = drink.lower().strip()
    
    # Define groups of similar drinks as specified in surveyContent.md
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

def clean_hot_sauce(text):
    """Q8: Process into vector of 5*1"""
    sauce_types = {
        'none': 0,
        'mild': 0,
        'medium': 0,
        'hot': 0,
        'with_sauce': 0
    }
    
    if not isinstance(text, str):
        return sauce_types
    
    text = text.lower()
    
    if 'none' in text:
        sauce_types['none'] = 1
    elif 'little' in text or 'mild' in text:
        sauce_types['mild'] = 1
    elif 'moderate' in text or 'medium' in text:
        sauce_types['medium'] = 1
    elif 'lot' in text or 'hot' in text:
        sauce_types['hot'] = 1
    elif 'with my hot sauce' in text:
        sauce_types['with_sauce'] = 1
    
    return sauce_types

def clean_data():
    """Clean the survey data according to the logic in surveyContent.md."""
    # Input file
    input_file = 'data/cleaned_data_combined.csv'
    
    # Output directories
    main_output_dir = 'data/improved_cleaned_data'
    
    # Create output directories if they don't exist
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, quotechar='"', escapechar='\\', 
                     on_bad_lines='skip', low_memory=False)
    
    print(f"Original data shape: {df.shape}")
    
    # Create a new DataFrame for cleaned data
    cleaned_data = pd.DataFrame()
    
    # Keep the id and label
    cleaned_data['id'] = df['id']
    cleaned_data['Label'] = df['Label']
    
    # Q1: Complexity (leave it as is)
    print("Processing Q1: Complexity...")
    cleaned_data['complexity'] = df['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)'].apply(clean_complexity)
    
    # Q2: Ingredients
    print("Processing Q2: Ingredients...")
    cleaned_data['ingredients'] = df['Q2: How many ingredients would you expect this food item to contain?'].apply(clean_ingredients)
    
    # Q3: Settings
    print("Processing Q3: Settings...")
    settings = df['Q3: In what setting would you expect this food to be served? Please check all that apply'].apply(extract_settings)
    settings_df = pd.DataFrame.from_records(settings.tolist())
    cleaned_data = pd.concat([cleaned_data, settings_df], axis=1)
    
    # Q4: Price
    print("Processing Q4: Price...")
    cleaned_data['price'] = df['Q4: How much would you expect to pay for one serving of this food item?'].apply(clean_price)
    
    # Q5: Movie
    print("Processing Q5: Movie...")
    cleaned_data['movie'] = df['Q5: What movie do you think of when thinking of this food item?'].apply(clean_movie)
    
    # Q6: Drink
    print("Processing Q6: Drinks...")
    
    # First normalize drinks
    drink_col = 'Q6: What drink would you pair with this food item?'
    df['normalized_drink'] = df[drink_col].apply(normalize_drink)
    
    # Count occurrences of each drink category
    drink_counts = Counter(df['normalized_drink'].dropna())
    print(f"Drink categories after normalization: {dict(drink_counts.most_common())}")
    
    # Get top 4 drinks
    top_drinks = [drink for drink, _ in drink_counts.most_common(4)]
    print(f"Top 4 drink categories: {top_drinks}")
    
    # Create one-hot encoding for top drinks
    for drink in top_drinks:
        cleaned_data[f'drink_{drink}'] = (df['normalized_drink'] == drink).astype(int)
    
    # Add 'other' category for non-top drinks
    cleaned_data['drink_other'] = (~df['normalized_drink'].isin(top_drinks)).astype(int)
    
    # Q7: Who Reminds
    print("Processing Q7: Who reminds of...")
    reminds = df['Q7: When you think about this food item, who does it remind you of?'].apply(extract_who_reminds)
    reminds_df = pd.DataFrame.from_records(reminds.tolist())
    cleaned_data = pd.concat([cleaned_data, reminds_df], axis=1)
    
    # Q8: Hot Sauce
    print("Processing Q8: Hot sauce...")
    sauce = df['Q8: How much hot sauce would you add to this food item?'].apply(clean_hot_sauce)
    sauce_df = pd.DataFrame.from_records(sauce.tolist())
    cleaned_data = pd.concat([cleaned_data, sauce_df], axis=1)
    
    # Calculate group medians for Q2 (ingredients) and Q4 (price)
    print("Calculating group medians for ingredients and price...")
    group_medians = {}
    for label in cleaned_data['Label'].unique():
        group_data = cleaned_data[cleaned_data['Label'] == label]
        group_medians[label] = {
            'ingredients': group_data['ingredients'].median(),
            'price': group_data['price'].median()
        }
        print(f"Group '{label}' medians - Ingredients: {group_medians[label]['ingredients']:.2f}, Price: {group_medians[label]['price']:.2f}")
    
    # Fill NULL values with group medians for Q2 and Q4
    print("Filling NULL values with group medians for Q2 and Q4...")
    for idx, row in cleaned_data.iterrows():
        label = row['Label']
        
        # Fill ingredients
        if pd.isna(row['ingredients']):
            cleaned_data.at[idx, 'ingredients'] = group_medians[label]['ingredients']
        
        # Fill price
        if pd.isna(row['price']):
            cleaned_data.at[idx, 'price'] = group_medians[label]['price']
    
    # Fill NULL values in Q5 (movie) with "other"
    print("Filling NULL values in movie with 'other'...")
    cleaned_data['movie'] = cleaned_data['movie'].fillna("other")
    
    # Save the main cleaned data
    output_file = os.path.join(main_output_dir, 'improved_survey_data.csv')
    print(f"Saving improved cleaned data to {output_file}...")
    cleaned_data.to_csv(output_file, index=False)
    
    # Extract each question to a separate file (same format as data/cleaned_data)
    print("Extracting individual questions to separate files...")
    
    # Define the questions and their corresponding columns
    questions = {
        'Q1_complexity': ['id', 'complexity', 'Label'],
        'Q2_ingredients': ['id', 'ingredients', 'Label'],
        'Q3_settings': ['id', 'weekday_lunch', 'weekday_dinner', 'weekend_lunch', 
                        'weekend_dinner', 'party', 'late_night', 'Label'],
        'Q4_price': ['id', 'price', 'Label'],
        'Q5_movie': ['id', 'movie', 'Label'],
        'Q6_drinks': ['id'] + [col for col in cleaned_data.columns if col.startswith('drink_')] + ['Label'],
        'Q7_who_reminds': ['id', 'parents', 'siblings', 'friends', 'teachers', 'strangers', 'Label'],
        'Q8_hot_sauce': ['id', 'none', 'mild', 'medium', 'hot', 'with_sauce', 'Label']
    }
    
    # Extract each question into a separate CSV file
    for q_id, columns in questions.items():
        # Check if all columns exist in the dataframe
        if all(col in cleaned_data.columns for col in columns):
            # Extract the columns
            output_df = cleaned_data[columns].copy()
            
            # Save to CSV
            output_file = os.path.join(main_output_dir, f'{q_id}.csv')
            output_df.to_csv(output_file, index=False)
            print(f"Extracted {q_id} to {output_file}")
        else:
            missing_cols = [col for col in columns if col not in cleaned_data.columns]
            print(f"Warning: Some columns for {q_id} not found in the dataset: {missing_cols}")
    
    print(f"Cleaning completed. Improved data shape: {cleaned_data.shape}")
    print(f"Columns in improved data: {cleaned_data.columns.tolist()}")
    
    # Count null values
    null_counts = cleaned_data.isnull().sum()
    print("\nNull values per column (should be 0 for ingredients, price, and movie):")
    for column, count in null_counts[null_counts > 0].items():
        print(f"  {column}: {count} nulls ({count/len(cleaned_data)*100:.2f}%)")
    
    # Count unique labels
    label_counts = cleaned_data['Label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(cleaned_data)*100:.2f}%)")

if __name__ == "__main__":
    clean_data() 