import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Create output directory for figures if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Load the dataset
drinks_df = pd.read_csv('./final_combined_data/Q6_drinks.csv')
print("Data loaded successfully.")

# First, let's examine the structure of the data
print("Q6 Drinks columns:", drinks_df.columns.tolist())

# Assuming the drink types are individual columns after 'id' and 'Label'
drink_columns = [col for col in drinks_df.columns if col not in ['id', 'Label']]
print(f"Drink columns: {drink_columns}")

# Create a dataframe to hold the percentages
drinks_percentages = pd.DataFrame()

# Calculate percentage of each drink for each food type
for food_type in ['Pizza', 'Shawarma', 'Sushi']:
    food_data = drinks_df[drinks_df['Label'] == food_type]
    for drink in drink_columns:
        percentage = food_data[drink].mean() * 100  # Convert to percentage
        drinks_percentages.loc[food_type, drink] = percentage

print("Drink percentages calculated.")

# 1. Create a heatmap to visualize drink preferences by food type
plt.figure(figsize=(14, 8))
sns.heatmap(drinks_percentages, annot=True, cmap='YlGnBu', fmt='.1f', 
            cbar_kws={'label': 'Percentage (%)'})
plt.title('Drink Preferences by Food Type')
plt.tight_layout()
plt.savefig('figures/Q6_drinks_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Create a grouped bar chart for drink preferences
drinks_melted = drinks_percentages.reset_index().melt(id_vars='index', 
                                                     var_name='Drink', 
                                                     value_name='Percentage')
drinks_melted.rename(columns={'index': 'Food Type'}, inplace=True)

plt.figure(figsize=(16, 10))
sns.barplot(x='Drink', y='Percentage', hue='Food Type', data=drinks_melted, palette='Set2')
plt.title('Drink Preferences by Food Type')
plt.xlabel('Drink Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Food Type')
plt.tight_layout()
plt.savefig('figures/Q6_drinks_barplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create a stacked bar chart to show proportion of drinks for each food type
# Transpose the data to have drinks as rows and food types as columns
drinks_proportion = drinks_percentages.copy()
# Normalize each row (food type) so they sum to 100%
for food_type in drinks_proportion.index:
    drinks_proportion.loc[food_type] = drinks_proportion.loc[food_type] / drinks_proportion.loc[food_type].sum() * 100

plt.figure(figsize=(12, 8))
drinks_proportion.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20')
plt.title('Proportion of Drink Preferences by Food Type')
plt.xlabel('Food Type')
plt.ylabel('Percentage (%)')
plt.legend(title='Drink Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/Q6_drinks_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Create a pie chart for each food type to show top drinks
plt.figure(figsize=(18, 6))

for i, food_type in enumerate(['Pizza', 'Shawarma', 'Sushi']):
    plt.subplot(1, 3, i+1)
    
    # Sort the drinks by percentage and get top 5 (plus 'Other' for the rest)
    drink_data = drinks_percentages.loc[food_type].sort_values(ascending=False)
    top_drinks = drink_data.head(5)
    
    # Add 'Other' category for the rest if there are more than 5 drinks
    if len(drink_data) > 5:
        top_drinks['Other'] = drink_data[5:].sum()
    
    # Create pie chart
    plt.pie(top_drinks, labels=top_drinks.index, autopct='%1.1f%%', 
            startangle=90, shadow=False, 
            colors=sns.color_palette('Set3', len(top_drinks)))
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Top Drinks Paired with {food_type}')

plt.tight_layout()
plt.savefig('figures/Q6_drinks_pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("Drink visualizations complete! Check the 'figures' directory for output files.") 