import pandas as pd
import os

def extract_headers(input_file, output_file):
    """
    Extract column names (header row) from a CSV file and save them to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file with headers
    """
    print(f"Reading headers from {input_file}...")
    
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Read just the first row to get headers (nrows=0 gives only headers)
        df = pd.read_csv(input_file, nrows=0)
        
        # Get the column names
        headers = df.columns.tolist()
        
        # Create a DataFrame with headers as a row (transpose)
        headers_df = pd.DataFrame([headers], columns=headers)
        
        # Also create a DataFrame with just the headers as a single column
        headers_list_df = pd.DataFrame(headers, columns=['Column_Name'])
        
        # Save to CSV
        headers_list_df.to_csv(output_file, index=False)
        
        # Also save a version with index numbers for reference
        headers_list_df.to_csv(output_file.replace('.csv', '_with_index.csv'), index=True)
        
        print(f"Successfully extracted {len(headers)} column names.")
        print(f"Headers saved to {output_file}")
        print(f"Headers with index numbers saved to {output_file.replace('.csv', '_with_index.csv')}")
        return True
    
    except Exception as e:
        print(f"Error extracting headers: {str(e)}")
        return False

if __name__ == "__main__":
    # Default file paths
    input_file = "final_data_with_bow.csv"
    output_file = "feature_names.csv"
    
    # Check if the default input file exists
    if not os.path.exists(input_file):
        # Try looking in the data directory
        alternate_input = os.path.join("data", "final_combined_data", "final_data_with_bow.csv")
        if os.path.exists(alternate_input):
            print(f"Using alternate input file: {alternate_input}")
            input_file = alternate_input
        else:
            # List available CSV files
            print(f"Warning: Default input file '{input_file}' not found.")
            print("Available CSV files:")
            for file in os.listdir("."):
                if file.endswith(".csv"):
                    print(f"  - {file}")
            if os.path.exists("data"):
                for root, dirs, files in os.walk("data"):
                    for file in files:
                        if file.endswith(".csv"):
                            print(f"  - {os.path.join(root, file)}")
            
            # Ask for input file path
            input_file = input("Enter the path to the input CSV file: ")
    
    # Extract headers
    extract_headers(input_file, output_file) 