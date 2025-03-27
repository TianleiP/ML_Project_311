import pandas as pd
import numpy as np
import re
import os
from collections import Counter, defaultdict

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

def create_movie_bow_features(movie_series):
    """Create bag-of-words features for movie responses."""
    # First clean the text
    cleaned_text = movie_series.apply(lambda x: clean_text(x) if pd.notna(x) else "")
    
    # Tokenize (split into words)
    tokenized = cleaned_text.apply(lambda x: x.split() if x else [])
    
    # Build vocabulary (all unique words)
    all_words = set()
    for tokens in tokenized:
        all_words.update(tokens)
    
    vocabulary = sorted(list(all_words))
    print(f"Movie vocabulary size: {len(vocabulary)} words")
    
    # Create document-term matrix
    bow_data = []
    for tokens in tokenized:
        # Count words in this document
        word_counts = defaultdict(int)
        for word in tokens:
            word_counts[word] += 1
        
        # Create sparse representation
        doc_features = {}
        for word in word_counts:
            if word in vocabulary:
                doc_features[f"movie_{word}"] = word_counts[word]
        
        bow_data.append(doc_features)
    
    # Convert to DataFrame (this will automatically fill missing values with 0)
    bow_df = pd.DataFrame(bow_data)
    bow_df.fillna(0, inplace=True)
    
    return bow_df, vocabulary

def process_final_data():
    """Process survey data with numeric hot sauce and movie bag-of-words."""
    # Input file
    input_file = 'data/cleaned_data_combined.csv'
    
    # Output directory
    output_dir = 'data/final_combined_data'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    movie_col = 'Q5: What movie do you think of when thinking of this food item?'
    cleaned_data['movie'] = df[movie_col].apply(clean_movie)
    
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
    
    # Q8: Hot Sauce (Numeric version 1-5)
    print("Processing Q8: Hot sauce as numeric...")
    cleaned_data['hot_sauce'] = df['Q8: How much hot sauce would you add to this food item?'].apply(clean_hot_sauce_numeric)
    
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
    
    # Calculate median hot sauce value per food type for missing values
    group_hot_sauce_medians = {}
    for label in cleaned_data['Label'].unique():
        group_data = cleaned_data[cleaned_data['Label'] == label]
        median_hot_sauce = group_data['hot_sauce'].median()
        group_hot_sauce_medians[label] = median_hot_sauce
        print(f"Group '{label}' median hot sauce: {median_hot_sauce:.2f}")
    
    # Fill NULL values in hot_sauce with group medians
    print("Filling NULL values in hot_sauce with group medians...")
    for idx, row in cleaned_data.iterrows():
        if pd.isna(row['hot_sauce']):
            label = row['Label']
            cleaned_data.at[idx, 'hot_sauce'] = group_hot_sauce_medians[label]
    
    # Fill NULL values in Q5 (movie) with "other"
    print("Filling NULL values in movie with 'other'...")
    cleaned_data['movie'] = cleaned_data['movie'].fillna("other")
    
    # Save the basic cleaned data
    basic_output_file = os.path.join(output_dir, 'cleaned_survey_data.csv')
    print(f"Saving basic cleaned data to {basic_output_file}...")
    cleaned_data.to_csv(basic_output_file, index=False)
    
    # Create a file with just the movie responses
    movie_data = pd.DataFrame({
        'id': cleaned_data['id'],
        'movie': cleaned_data['movie'],
        'Label': cleaned_data['Label']
    })
    movie_output_file = os.path.join(output_dir, 'Q5_movie_responses.csv')
    print(f"Saving movie responses to {movie_output_file}...")
    movie_data.to_csv(movie_output_file, index=False)
    
    # Create bag-of-words features for movie data
    print("Creating bag-of-words features for movie data...")
    bow_df, vocabulary = create_movie_bow_features(cleaned_data['movie'])
    
    # Save vocabulary to a file
    vocab_df = pd.DataFrame({
        'word': vocabulary,
        'index': range(len(vocabulary))
    })
    vocab_output_file = os.path.join(output_dir, 'Q5_movie_vocabulary.csv')
    print(f"Saving vocabulary to {vocab_output_file}...")
    vocab_df.to_csv(vocab_output_file, index=False)
    
    # Now create the complete dataset (with movies as bag-of-words)
    # First, remove the original movie text column
    final_data = cleaned_data.drop(columns=['movie'])
    
    # Then add the bag-of-words features
    print("Adding bag-of-words features to final dataset...")
    for col in bow_df.columns:
        final_data[col] = bow_df[col].values
    
    # Save the complete data
    final_output_file = os.path.join(output_dir, 'final_data_with_bow.csv')
    print(f"Saving final data to {final_output_file}...")
    final_data.to_csv(final_output_file, index=False)
    
    # Extract each question to separate files
    print("Extracting individual questions to separate files...")
    
    # Define the questions and their corresponding columns
    questions = {
        'Q1_complexity': ['id', 'complexity', 'Label'],
        'Q2_ingredients': ['id', 'ingredients', 'Label'],
        'Q3_settings': ['id', 'weekday_lunch', 'weekday_dinner', 'weekend_lunch', 
                         'weekend_dinner', 'party', 'late_night', 'Label'],
        'Q4_price': ['id', 'price', 'Label'],
        'Q5_movie_bow': ['id'] + list(bow_df.columns) + ['Label'],
        'Q6_drinks': ['id'] + [col for col in cleaned_data.columns if col.startswith('drink_')] + ['Label'],
        'Q7_who_reminds': ['id', 'parents', 'siblings', 'friends', 'teachers', 'strangers', 'Label'],
        'Q8_hot_sauce': ['id', 'hot_sauce', 'Label']
    }
    
    # Extract Q5 bag-of-words to a separate file (with Label)
    q5_bow_df = pd.DataFrame({'id': cleaned_data['id']})
    q5_bow_df = pd.concat([q5_bow_df, bow_df], axis=1)
    q5_bow_df['Label'] = cleaned_data['Label']
    q5_bow_output_file = os.path.join(output_dir, 'Q5_movie_bow.csv')
    print(f"Saving movie bag-of-words to {q5_bow_output_file}...")
    q5_bow_df.to_csv(q5_bow_output_file, index=False)
    
    # Extract other questions to separate files
    for q_id, columns in questions.items():
        if q_id != 'Q5_movie_bow':  # We already saved Q5 bag-of-words above
            # Create a temporary dataframe for each question
            if q_id.startswith('Q5'):
                # Skip Q5 as we've already handled it
                continue
                
            # For other questions, extract from cleaned_data
            valid_columns = [col for col in columns if col in final_data.columns or col == 'id' or col == 'Label']
            if len(valid_columns) >= 3:  # At least id, one feature, and Label
                output_df = final_data[valid_columns].copy()
                
                # Save to CSV
                output_file = os.path.join(output_dir, f'{q_id}.csv')
                output_df.to_csv(output_file, index=False)
                print(f"Extracted {q_id} to {output_file}")
    
    print(f"Basic cleaned data shape: {cleaned_data.shape}")
    print(f"Movie bag-of-words shape: {bow_df.shape}")
    print(f"Final data with bag-of-words shape: {final_data.shape}")
    print(f"Vocabulary size: {len(vocabulary)} words")
    
    # Count null values in the final dataset
    null_counts = final_data.isnull().sum()
    print("\nNull values per column (should be 0 for ingredients, price, hot_sauce):")
    for column, count in null_counts[null_counts > 0].items():
        print(f"  {column}: {count} nulls ({count/len(final_data)*100:.2f}%)")
    
    # Count unique labels
    label_counts = final_data['Label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(final_data)*100:.2f}%)")
    
    print("\nProcessing completed successfully!")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    process_final_data() 