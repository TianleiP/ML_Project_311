import pandas as pd
import re
from collections import Counter

def normalize_drink_name(drink):
    """Normalize drink names by combining similar terms."""
    drink = drink.lower().strip()
    
    # Define groups of similar drinks
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
            
    return drink

def main():
    # Input and output files
    input_file = 'data/freeform_analysis/Q6_responses.csv'
    output_file = 'data/freeform_analysis/combined_top_drinks.csv'
    
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
                    normalized_drink = normalize_drink_name(d)
                    drinks.append(normalized_drink)
    
    # Count drink occurrences
    drink_counter = Counter(drinks)
    
    # Get top drinks (let's get more than 20 initially as some will be combined)
    top_drinks = drink_counter.most_common(30)
    
    # Convert to DataFrame
    top_drinks_df = pd.DataFrame(top_drinks, columns=['Drink', 'Count'])
    
    # Calculate percentage
    total_mentions = sum(drink_counter.values())
    top_drinks_df['Percentage'] = (top_drinks_df['Count'] / total_mentions * 100).round(2)
    
    # Add rank column
    top_drinks_df.insert(0, 'Rank', range(1, len(top_drinks_df) + 1))
    
    # Sort by count and take top 20
    top_drinks_df = top_drinks_df.head(20)
    
    # Save to CSV
    top_drinks_df.to_csv(output_file, index=False)
    
    print(f"Combined top drinks saved to {output_file}")
    print("\nCombined drink categories:")
    print("1. coca cola: coke, coca cola, diet coke, cola, coke zero, pepsi, diet pepsi")
    print("2. tea: tea, green tea, iced tea, ice tea, bubble tea, milk tea")
    print("3. water: water, sparkling water, mineral water, soda water")
    print("4. beer: beer, craft beer, cold beer")
    print("5. juice: juice, orange juice, apple juice, fruit juice, lemon juice")
    print("6. soda: soda, pop, soft drink, carbonated drink, sprite, fanta, 7up, mountain dew")
    print("7. wine: wine, red wine, white wine, rose wine")
    print("8. coffee: coffee, iced coffee, espresso, latte")
    print("9. milk: milk, chocolate milk, dairy")
    
    print("\nTop 10 combined drinks:")
    print(top_drinks_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main() 