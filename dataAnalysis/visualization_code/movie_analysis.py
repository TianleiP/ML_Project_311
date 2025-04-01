import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Create output directory for figures if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load the dataset
movie_df = pd.read_csv('./final_combined_data/Q5_movie_responses.csv')
print("Data loaded successfully.")

# First, let's examine the structure of the data
print("Q5 Movie responses columns:", movie_df.columns.tolist())
print(f"Total responses: {len(movie_df)}")

# Function to check if a string contains "avengers" (case-insensitive)
def contains_avengers(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'avengers', str(text).lower()))

# Add a column indicating if the response contains "avengers"
movie_df['contains_avengers'] = movie_df['movie'].apply(contains_avengers)

# Group by food type and count "avengers" mentions
avengers_count = movie_df.groupby('Label')['contains_avengers'].sum().reset_index()
avengers_count.rename(columns={'contains_avengers': 'Avengers Mentions'}, inplace=True)

# Calculate percentage of mentions within each food type
food_type_counts = movie_df.groupby('Label').size().reset_index(name='Total Responses')
avengers_percentage = pd.merge(avengers_count, food_type_counts, on='Label')
avengers_percentage['Percentage'] = (avengers_percentage['Avengers Mentions'] / 
                                   avengers_percentage['Total Responses'] * 100)

print("Avengers mentions by food type:")
print(avengers_percentage)

# Visualize the counts with a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Label', y='Avengers Mentions', data=avengers_count, palette='Set2')
plt.title('Number of "Avengers" Mentions in Movie Responses by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Count of "Avengers" Mentions')
for i, count in enumerate(avengers_count['Avengers Mentions']):
    plt.text(i, count + 0.1, str(count), ha='center')
plt.tight_layout()
plt.savefig('figures/Q5_avengers_counts.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize the percentage with a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Label', y='Percentage', data=avengers_percentage, palette='Set3')
plt.title('Percentage of "Avengers" Mentions in Movie Responses by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Percentage of Responses (%)')
for i, pct in enumerate(avengers_percentage['Percentage']):
    plt.text(i, pct + 0.1, f"{pct:.1f}%", ha='center')
plt.tight_layout()
plt.savefig('figures/Q5_avengers_percentage.png', dpi=300, bbox_inches='tight')
plt.close()

# Bonus: Find most common movie responses for each food type
def extract_common_movies(food_type, top_n=5):
    # Filter for the specific food type
    food_data = movie_df[movie_df['Label'] == food_type]
    
    # Count occurrences of each movie (excluding NaN)
    movie_counts = food_data['movie'].dropna().value_counts().head(top_n)
    
    return movie_counts

# Get top movies for each food type
pizza_movies = extract_common_movies('Pizza')
shawarma_movies = extract_common_movies('Shawarma')
sushi_movies = extract_common_movies('Sushi')

print("\nTop movies mentioned for Pizza:")
print(pizza_movies)

print("\nTop movies mentioned for Shawarma:")
print(shawarma_movies)

print("\nTop movies mentioned for Sushi:")
print(sushi_movies)

# Find all Avengers-related responses for Shawarma (we expect more here)
shawarma_avengers = movie_df[(movie_df['Label'] == 'Shawarma') & 
                             (movie_df['contains_avengers'] == True)]

print(f"\nAvengers-related responses for Shawarma ({len(shawarma_avengers)}):")
for _, row in shawarma_avengers.iterrows():
    print(f"- {row['movie']}")

print("\nAnalysis complete!") 