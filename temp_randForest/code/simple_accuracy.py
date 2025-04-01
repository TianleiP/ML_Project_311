import csv

def calculate_accuracy():
    # File paths
    predictions_file = "predictions_aligned.csv"
    actual_file = "cleaned_data_combined_modified.csv"
    
    # Counters
    correct = 0
    total = 0
    
    # Open both files with UTF-8 encoding
    with open(predictions_file, 'r', encoding='utf-8') as pred_f, open(actual_file, 'r', encoding='utf-8') as actual_f:
        pred_reader = csv.reader(pred_f)
        actual_reader = csv.reader(actual_f)
        
        # Skip headers
        pred_header = next(pred_reader)
        actual_header = next(actual_reader)
        
        # Find index of prediction and label columns
        pred_idx = pred_header.index('prediction')
        
        # Find Label column in actual file
        try:
            label_idx = actual_header.index('Label')
        except ValueError:
            print(f"Error: 'Label' column not found in {actual_file}")
            print(f"Available columns: {actual_header}")
            return
        
        # Compare line by line
        for i, (pred_row, actual_row) in enumerate(zip(pred_reader, actual_reader), 1):
            if len(pred_row) <= pred_idx or len(actual_row) <= label_idx:
                print(f"Warning: Line {i} has invalid format, skipping")
                continue
            
            prediction = pred_row[pred_idx]
            actual_label = actual_row[label_idx]
            
            # Compare
            if prediction == actual_label:
                correct += 1
            
            total += 1
            
            # Print progress every 100 lines
            if i % 100 == 0:
                print(f"Processed {i} lines...")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Print results
    print(f"\nResults:")
    print(f"Total lines compared: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    calculate_accuracy() 