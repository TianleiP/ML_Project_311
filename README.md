# Food Classification Project

This project aims to classify food items (Pizza, Shawarma, or Sushi) based on survey responses about food characteristics.

## Project Structure

### Root Directory
- `final_process_combined.py`: Script for cleaning and preprocessing survey data according to the logic in surveyContent.md.
- `example_pred.py`: Example prediction script showing the required function structure for submissions.
- `challenge_basic.py`: Basic implementation of kNN using sklearn (for reference only).
- `README.md`: This file, explaining the project structure.

### /instructions
- `projectInstruction.md`: Project requirements and submission guidelines.
- `modelsLearnedSoFar.md`: List of machine learning models covered in the course and their suitability.
- `surveyContent.md`: Details about survey questions, answer formats, and cleaning logic.

### /helper_files
Contains utility scripts for data processing.

### /data
- `cleaned_data_combined.csv`: Original survey data with responses about food items.
- `cleaned_data_combined_modified.csv`: Modified version of the original data.

#### /data/freeform_analysis
Contains extracted raw responses and analyses for free-form questions.

#### /data/cleaned_data
Contains the main cleaned dataset and individual question data.

#### /data/improved_cleaned_data
Contains improved versions of the cleaned data.

#### /data/final_combined_data
Contains processed final datasets.



## Data Processing Pipeline

1. **Raw Data Exploration**: Extract and analyze free-form question responses
2. **Data Cleaning**: Process survey questions according to the cleaning logic in surveyContent.md
3. **Feature Engineering**: Convert responses into numerical and categorical features suitable for machine learning

## Project Goal

The final deliverable is a `pred.py` script that can predict food types without using sklearn, PyTorch, or TensorFlow, while maintaining high accuracy. The script must:

1. Include a `predict_all` function that takes a CSV filename as input
2. Return predictions for each data point (Pizza, Shawarma, or Sushi)
3. Use only basic Python libraries (numpy, pandas, csv, etc.)
4. Run efficiently within the specified resource constraints

## Suitable Models for Implementation

- K-Nearest Neighbors
- Decision Tree
- Logistic Regression
- Naive Bayes
- Gaussian Discriminant Analysis

## Requirements

- Python 3.10 or compatible
- Required packages: numpy, pandas
- No use of sklearn, PyTorch, or TensorFlow in the final submission
