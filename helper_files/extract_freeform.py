import pandas as pd
import os

def main():
    # Read the original CSV file
    input_file = 'data/cleaned_data_combined.csv'
    output_dir = 'data/freeform_analysis'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(input_file, quotechar='"', escapechar='\\', 
                     error_bad_lines=False, warn_bad_lines=True, 
                     low_memory=False)
    
    # Define the free-form questions
    freeform_questions = {
        'Q2': 'Q2: How many ingredients would you expect this food item to contain?',
        'Q4': 'Q4: How much would you expect to pay for one serving of this food item?',
        'Q5': 'Q5: What movie do you think of when thinking of this food item?',
        'Q6': 'Q6: What drink would you pair with this food item?'
    }
    
    # Extract each free-form question into a separate CSV file
    for q_id, q_column in freeform_questions.items():
        if q_column in df.columns:
            # Extract ID, question response, and Label
            output_df = df[['id', q_column, 'Label']].copy()
            
            # Save to CSV
            output_file = os.path.join(output_dir, f'{q_id}_responses.csv')
            output_df.to_csv(output_file, index=False)
            print(f"Extracted {q_id} responses to {output_file}")
        else:
            print(f"Warning: Column '{q_column}' not found in the dataset.")
    
    print("Extraction completed!")

if __name__ == "__main__":
    main() 