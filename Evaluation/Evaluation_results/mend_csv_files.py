import pandas as pd
import os


import pandas as pd
import os

# Files to rename column header
files_to_rename = [
    "MAPPO_evaluation_results_K3_cpos0.5_cphi0.1_n10000_dir0_jam0.csv",
    "MAPPO_evaluation_results_K5_cpos0.5_cphi0.1_n10000_dir0_jam0.csv"
]

base_dir = "Evaluation/MARL_evaluations/MAPPO"

for filename in files_to_rename:
    file_path = os.path.join(base_dir, filename)
    
    if os.path.exists(file_path):
        print(f"Processing {filename}...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'sum_distance' column exists
        if 'sum_distance' in df.columns:
            # Rename the column
            df = df.rename(columns={'sum_distance': 'agent_sum_distance'})
            
            # Save the modified dataframe back to the CSV file
            df.to_csv(file_path, index=False)
            
            print(f"  Successfully renamed 'sum_distance' to 'agent_sum_distance'")
            print(f"  New columns: {list(df.columns)}")
        else:
            print(f"  Warning: Column 'sum_distance' not found in {filename}")
            print(f"  Available columns: {list(df.columns)}")
    else:
        print(f"  File not found: {file_path}")

print("\nColumn renaming complete!")


"""
# Define the problematic entries
problematic_entries = {
    1: 5393,
    5: 7865,
    7: 1411
}

# Base directory for MADDPG evaluations
base_dir = "Evaluation/MARL_evaluations/MADDPG"

# Process each K value
for k, row_idx in problematic_entries.items():
    # Construct the file path (adjust pattern based on your actual filenames)
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and f'K{k}' in f]
    
    for csv_file in csv_files:
        file_path = os.path.join(base_dir, csv_file)
        
        print(f"Processing {file_path}...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the row index exists
        if row_idx < len(df):
            print(f"  Removing row {row_idx} from {csv_file}")
            
            # Remove the row at the specified index
            df = df.drop(row_idx).reset_index(drop=True)
            
            # Save the modified dataframe back to the CSV file
            df.to_csv(file_path, index=False)
            
            print(f"  Successfully removed row {row_idx}. New file length: {len(df)}")
        else:
            print(f"  Warning: Row {row_idx} not found in {csv_file} (file length: {len(df)})")

print("\nAll problematic rows have been removed and data has been shifted!")
"""