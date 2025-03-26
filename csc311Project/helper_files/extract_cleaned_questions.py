import pandas as pd
import os

def main():
    # Input file and output directory
    input_file = 'data/cleaned_survey_data.csv'
    output_dir = 'data/cleaned_question_extracts'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the cleaned CSV file
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Define the questions and their corresponding columns
    questions = {
        'Q1_complexity': ['id', 'complexity', 'Label'],
        'Q2_ingredients': ['id', 'ingredients', 'Label'],
        'Q3_settings': ['id', 'weekday_lunch', 'weekday_dinner', 'weekend_lunch', 
                        'weekend_dinner', 'party', 'late_night', 'Label'],
        'Q4_price': ['id', 'price', 'Label'],
        'Q5_movie': ['id', 'movie', 'Label'],
        'Q6_drinks': ['id'] + [col for col in df.columns if col.startswith('drink_')] + ['Label'],
        'Q7_who_reminds': ['id', 'parents', 'siblings', 'friends', 'teachers', 'strangers', 'Label'],
        'Q8_hot_sauce': ['id', 'none', 'mild', 'medium', 'hot', 'with_sauce', 'Label']
    }
    
    # Extract each question into a separate CSV file
    for q_id, columns in questions.items():
        # Check if all columns exist in the dataframe
        if all(col in df.columns for col in columns):
            # Extract the columns
            output_df = df[columns].copy()
            
            # Save to CSV
            output_file = os.path.join(output_dir, f'{q_id}.csv')
            output_df.to_csv(output_file, index=False)
            print(f"Extracted {q_id} to {output_file} with columns: {columns}")
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            print(f"Warning: Some columns for {q_id} not found in the dataset: {missing_cols}")
    
    print("Extraction completed!")

if __name__ == "__main__":
    main() 