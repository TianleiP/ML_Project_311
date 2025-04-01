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

# Load the datasets
settings_df = pd.read_csv('./final_combined_data/Q3_settings.csv')
who_reminds_df = pd.read_csv('./final_combined_data/Q7_who_reminds.csv')

print("Data loaded successfully.")

# 1. Visualize Q3: Settings where food is served
# First, let's examine the structure of the data
print("Q3 Settings columns:", settings_df.columns.tolist())

# Assuming the settings are individual columns after 'id' and 'Label'
setting_columns = [col for col in settings_df.columns if col not in ['id', 'Label']]

# Create a dataframe to hold the percentages
settings_percentages = pd.DataFrame()

# Calculate percentage of each setting for each food type
for food_type in ['Pizza', 'Shawarma', 'Sushi']:
    food_data = settings_df[settings_df['Label'] == food_type]
    for setting in setting_columns:
        percentage = food_data[setting].mean() * 100  # Convert to percentage
        settings_percentages.loc[food_type, setting] = percentage

# Plot heatmap for settings
plt.figure(figsize=(12, 8))
sns.heatmap(settings_percentages, annot=True, cmap='YlGnBu', fmt='.1f', 
            cbar_kws={'label': 'Percentage (%)'})
plt.title('Settings Where Food Types Are Served')
plt.tight_layout()
plt.savefig('figures/Q3_settings_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Alternative: Create a grouped bar chart for settings
settings_melted = settings_percentages.reset_index().melt(id_vars='index', 
                                                          var_name='Setting', 
                                                          value_name='Percentage')
settings_melted.rename(columns={'index': 'Food Type'}, inplace=True)

plt.figure(figsize=(14, 8))
sns.barplot(x='Setting', y='Percentage', hue='Food Type', data=settings_melted, palette='Set2')
plt.title('Settings Where Food Types Are Served')
plt.xlabel('Setting')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Food Type')
plt.tight_layout()
plt.savefig('figures/Q3_settings_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q3 settings visualizations.")

# 2. Visualize Q7: Who food reminds of
# First, let's examine the structure of the data
print("Q7 Who Reminds columns:", who_reminds_df.columns.tolist())

# Assuming the person types are individual columns after 'id' and 'Label'
who_columns = [col for col in who_reminds_df.columns if col not in ['id', 'Label']]

# Create a dataframe to hold the percentages
who_percentages = pd.DataFrame()

# Calculate percentage of each person type for each food type
for food_type in ['Pizza', 'Shawarma', 'Sushi']:
    food_data = who_reminds_df[who_reminds_df['Label'] == food_type]
    for who in who_columns:
        percentage = food_data[who].mean() * 100  # Convert to percentage
        who_percentages.loc[food_type, who] = percentage

# Plot heatmap for who reminds
plt.figure(figsize=(12, 8))
sns.heatmap(who_percentages, annot=True, cmap='YlOrRd', fmt='.1f', 
            cbar_kws={'label': 'Percentage (%)'})
plt.title('Who Each Food Type Reminds People Of')
plt.tight_layout()
plt.savefig('figures/Q7_who_reminds_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Alternative: Create a grouped bar chart for who reminds
who_melted = who_percentages.reset_index().melt(id_vars='index', 
                                              var_name='Person Type', 
                                              value_name='Percentage')
who_melted.rename(columns={'index': 'Food Type'}, inplace=True)

plt.figure(figsize=(12, 8))
sns.barplot(x='Person Type', y='Percentage', hue='Food Type', data=who_melted, palette='Set3')
plt.title('Who Each Food Type Reminds People Of')
plt.xlabel('Person Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Food Type')
plt.tight_layout()
plt.savefig('figures/Q7_who_reminds_barplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created Q7 who reminds visualizations.")

print("Visualizations complete! Check the 'figures' directory for output files.") 