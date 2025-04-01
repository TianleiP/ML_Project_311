import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Create output directory for figures if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load the datasets directly from final_combined_data
complexity_df = pd.read_csv('./final_combined_data/Q1_complexity.csv')
ingredients_df = pd.read_csv('./final_combined_data/Q2_ingredients.csv')
price_df = pd.read_csv('./final_combined_data/Q4_price.csv')
hot_sauce_df = pd.read_csv('./final_combined_data/Q8_hot_sauce.csv')

print("Data loaded successfully.")

# 1. Q1: Complexity boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='complexity', data=complexity_df)
plt.title('Distribution of Food Complexity by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Complexity (1-5 scale)')
plt.savefig('figures/Q1_complexity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q1 complexity boxplot.")

# 2. Q2: Ingredients histograms
plt.figure(figsize=(12, 8))
food_types = ['Pizza', 'Shawarma', 'Sushi']
colors = ['#ff9999', '#66b3ff', '#99ff99']

for i, food_type in enumerate(food_types):
    food_data = ingredients_df[ingredients_df['Label'] == food_type]['ingredients']
    sns.histplot(food_data, kde=True, color=colors[i], label=food_type, alpha=0.6)

plt.title('Distribution of Ingredient Counts by Food Type')
plt.xlabel('Number of Ingredients')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('figures/Q2_ingredients_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q2 ingredients histogram.")

# 3. Q4: Price distributions
plt.figure(figsize=(12, 8))

# Remove outliers for better visualization (prices above 95th percentile)
price_threshold = price_df['price'].quantile(0.95)
filtered_price_df = price_df[price_df['price'] <= price_threshold]

sns.violinplot(x='Label', y='price', data=filtered_price_df, palette='pastel')
plt.title('Distribution of Price by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Price ($)')
plt.savefig('figures/Q4_price_violinplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q4 price violinplot.")

# 4. Q8: Hot sauce preferences
plt.figure(figsize=(12, 8))

# Create hot sauce count table
hot_sauce_counts = pd.crosstab(
    hot_sauce_df['Label'], 
    hot_sauce_df['hot_sauce'],
    normalize='index'
) * 100  # Convert to percentages

# Map numeric values to labels for clarity
hot_sauce_labels = {
    1: 'None',
    2: 'A little',
    3: 'Moderate',
    4: 'A lot',
    5: 'Food with hot sauce'
}
hot_sauce_counts.columns = [hot_sauce_labels.get(col, col) for col in hot_sauce_counts.columns]

# Plot stacked bar chart
hot_sauce_counts.plot(kind='bar', stacked=True, figsize=(12, 8), 
                     colormap='RdYlGn_r')
plt.title('Hot Sauce Preferences by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Percentage (%)')
plt.legend(title='Hot Sauce Amount')
plt.xticks(rotation=0)
plt.savefig('figures/Q8_hot_sauce_barchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q8 hot sauce barchart.")

# Generate summary statistics for the report
print("Generating summary statistics...")

# Complexity stats
complexity_stats = complexity_df.groupby('Label')['complexity'].describe()
complexity_stats.to_csv('figures/Q1_complexity_stats.csv')
print(f"Complexity statistics by food type:\n{complexity_stats}")

# Ingredients stats
ingredients_stats = ingredients_df.groupby('Label')['ingredients'].describe()
ingredients_stats.to_csv('figures/Q2_ingredients_stats.csv')
print(f"Ingredients statistics by food type:\n{ingredients_stats}")

# Price stats
price_stats = price_df.groupby('Label')['price'].describe()
price_stats.to_csv('figures/Q4_price_stats.csv')
print(f"Price statistics by food type:\n{price_stats}")

print("Visualizations complete! Check the 'figures' directory for output files.") 