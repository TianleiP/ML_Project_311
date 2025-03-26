# Food Classification Project

This project aims to classify food items (Pizza, Shawarma, or Sushi) based on survey responses about food characteristics.

## Project Structure

### Root Directory
- `clean_survey_data.py`: Main script to clean and preprocess survey data according to the logic in surveyContent.md.
- `example_pred.py`: Example prediction script provided as starter code, showing required function structure.
- `challenge_basic.py`: Basic implementation of kNN using sklearn (for reference only).
- `README.txt`: This file, explaining the project structure.

### /helper_files
- `extract_freeform.py`: Script to extract free-form question responses for separate analysis.
- `analyze_drinks.py`: Script to analyze drink preferences and generate top 20 mentioned drinks.
- `combine_drinks.py`: Script to combine similar drink categories and generate aggregated statistics.
- `extract_cleaned_questions.py`: Script to extract each question from cleaned data into separate files.

### /data
- `cleaned_data_combined.csv`: Original survey data with responses about food items.
- `cleaned_data_combined_modified.csv`: Modified version of the original data.

### /data/freeform_analysis
- `Q2_responses.csv`: Extracted raw responses about expected number of ingredients.
- `Q4_responses.csv`: Extracted raw responses about expected price.
- `Q5_responses.csv`: Extracted raw responses about movie associations.
- `Q6_responses.csv`: Extracted raw responses about drink pairings.
- `top_mentioned_drinks.csv`: Analysis of top 20 mentioned drinks from Q6 responses.
- `combined_top_drinks.csv`: Analysis of drinks after combining similar categories.

### /data/cleaned_data
- `cleaned_survey_data.csv`: Main cleaned dataset with all questions processed according to cleaning logic.
- `Q1_complexity.csv`: Cleaned complexity rating data (1-5 scale).
- `Q2_ingredients.csv`: Cleaned ingredient count data.
- `Q3_settings.csv`: Cleaned setting preferences as binary features.
- `Q4_price.csv`: Cleaned price data.
- `Q5_movie.csv`: Cleaned movie data (limited to ≤8 words).
- `Q6_drinks.csv`: Cleaned drink data with top categories as binary features.
- `Q7_who_reminds.csv`: Cleaned data about who the food reminds of.
- `Q8_hot_sauce.csv`: Cleaned hot sauce preference data.

### /instructions
- `instruction.md`: Project requirements and submission guidelines.
- `modelsLearnedSoFar.md`: List of machine learning models covered in the course.
- `surveyContent.md`: Details about survey questions, answer formats, and cleaning logic.

## Data Processing Pipeline

1. **Raw Data Exploration**: 
   - Extract and analyze free-form question responses with `extract_freeform.py`
   - Analyze drink preferences with `analyze_drinks.py` and `combine_drinks.py`

2. **Data Cleaning**:
   - Process all survey questions according to the logic in surveyContent.md using `clean_survey_data.py`
   - Extract individual cleaned question data with `extract_cleaned_questions.py`

3. **Next Steps**:
   - Implement machine learning models using cleaned data
   - Create final `pred.py` implementation using the best performing model

## Cleaning Logic Highlights

- **Numeric responses (Q1, Q2, Q4)**: Processed to standardized numeric values
- **Categorical responses (Q3, Q7, Q8)**: Converted to binary feature vectors
- **Text responses (Q5)**: Filtered by length (≤8 words)
- **Drink responses (Q6)**: Categorized and converted to binary features for top categories

## Requirements

- Python 3.10 or compatible
- Required packages: numpy, pandas (Additional packages like scikit-learn may be needed for development but not for the final solution)

## Project Goal

The final deliverable is a `pred.py` script that can predict food types without using sklearn, PyTorch, or TensorFlow, while maintaining high accuracy. 
