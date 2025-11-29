import pandas as pd


import os


def remove_column_from_csv(file_path, column_name):
    """
    Removes a specified column from a CSV file and saves the updated file.

    Parameters:
        file_path (str): Path to the CSV file.
        column_name (str): Name of the column to remove.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the column exists
        if column_name in df.columns:
            # Drop the specified column
            df = df.drop(columns=[column_name])

            # Save the updated DataFrame back to the same file
            df.to_csv(file_path, index=False)
            print(f"Column '{column_name}' removed successfully from {file_path}.")
        else:
            print(f"Column '{column_name}' does not exist in {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def modify_csv(file_path):
    """
    Removes the 'delivery_time' column and renames the 'directed_transmission' column to 'delivery_time'.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Remove the 'delivery_time' column if it exists
        if 'delivery_time' in df.columns:
            df = df.drop(columns=['delivery_time'])

        # Rename the 'directed_transmission' column to 'delivery_time' if it exists
        if 'directed_transmission' in df.columns:
            df = df.rename(columns={'directed_transmission': 'delivery_time'})

        # Save the updated DataFrame back to the same file
        df.to_csv(file_path, index=False)
        print(f"Modified CSV file successfully: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# remove_column_from_csv("MADDPG_evaluation_results_K1_cpos0.5_cphi0.1_n10000_dir0_jam1.csv", "message_air_distance")

#csv_path = "Evaluation/Evaluation_results/dir0_jam0_cpos0.5_cphi0.1/MADDPG/MADDPG_evaluation_results_K5_cpos0.5_cphi0.1_n10000_dir0_jam0.csv"
#remove_column_from_csv(csv_path, "message_air_distance")

# Example usage
modify_csv("c:/Users/Admin/OneDrive - Chalmers/Documents/Industridoktorand/Code/Information-Relaying-Final/Evaluation/Evaluation_results/dir0_jam1_cpos0.5_cphi0.1/MADDPG/MADDPG_evaluation_results_K1_cpos0.5_cphi0.1_n10000_dir0_jam1.csv")

