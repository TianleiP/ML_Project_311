import pandas as pd
import re
from collections import Counter

def main():
    # Input and output files
    input_file = 'data/freeform_analysis/Q6_responses.csv'
    output_file = 'data/freeform_analysis/top_drinks.csv'
    
    # Read the Q6 responses
    df = pd.read_csv(input_file)
    
    # Extract the drink column
    drink_column = 'Q6: What drink would you pair with this food item?'
    
    # Clean and normalize drink texts
    drinks = []
    for drink in df[drink_column]:
        if isinstance(drink, str):
            # Convert to lowercase and strip whitespace
            drink = drink.lower().strip()
            
            # Remove punctuation and extra spaces
            drink = re.sub(r'[^\w\s]', ' ', drink)
            drink = re.sub(r'\s+', ' ', drink).strip()
            
            # Split by common separators to handle multiple drinks in one response
            for d in re.split(r'[,&/]|\bor\b|\band\b|\bwith\b', drink):
                d = d.strip()
                if d:  # Only add non-empty strings
                    drinks.append(d)
    
    # Count drink occurrences
    drink_counter = Counter(drinks)
    
    # Get top 20 drinks
    top_drinks = drink_counter.most_common(20)
    
    # Convert to DataFrame
    top_drinks_df = pd.DataFrame(top_drinks, columns=['Drink', 'Count'])
    
    # Calculate percentage
    total_mentions = sum(drink_counter.values())
    top_drinks_df['Percentage'] = (top_drinks_df['Count'] / total_mentions * 100).round(2)
    
    # Add rank column
    top_drinks_df.insert(0, 'Rank', range(1, len(top_drinks_df) + 1))
    
    # Save to CSV
    top_drinks_df.to_csv(output_file, index=False)
    
    print(f"Top 20 drinks saved to {output_file}")
    print("\nTop 10 drinks:")
    print(top_drinks_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main() 