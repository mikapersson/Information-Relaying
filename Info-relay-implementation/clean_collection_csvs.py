import os
import glob
import pandas as pd

# ====== Configuration ======
SCENARIO              = "5_agent_one_way_not_random"   # Folder under simulation_data/
PROCESS_COLLECTION    = True                         # Remove duplicates in collection files?
PROCESS_EVAL          = True                         # Remove duplicates in eval files?
INPLACE               = True                         # Overwrite originals? If False, writes *_deduped.csv
TRUNCATE_AFTER_INDEX  = 2000                         # Set to an integer to truncate rows after this index (e.g., 100)
# ===========================

# Filename patterns for each mode
PATTERNS = {
    "collection": "collection_agent_reward_episode_reward_*.csv",
    "eval":       "eval_agent_reward_episode_reward_*.csv",
}

def dedupe_csv_file(path, inplace=True):
    """
    Read `path` (no header), drop the first occurrence of any duplicated
    values in column 0, keeping the last, and write back.
    """
    df = pd.read_csv(path, header=None)
    df_deduped = df.drop_duplicates(subset=0, keep='last')
    
    if inplace:
        df_deduped.to_csv(path, index=False, header=False)
    else:
        base, ext = os.path.splitext(path)
        outpath = f"{base}_deduped{ext}"
        df_deduped.to_csv(outpath, index=False, header=False)
    
    removed = len(df) - len(df_deduped)
    print(f"{os.path.basename(path)}: removed {removed} duplicate rows")

def truncate_csv_file(path, index, inplace=True):
    """
    Truncate CSV file by removing all rows after a specified index (exclusive).
    """
    df = pd.read_csv(path, header=None)
    if index < 0 or index > len(df):
        print(f"{os.path.basename(path)}: index {index} out of bounds, skipping.")
        return
    df_truncated = df.iloc[:index]

    if inplace:
        df_truncated.to_csv(path, index=False, header=False)
    else:
        base, ext = os.path.splitext(path)
        outpath = f"{base}_truncated{ext}"
        df_truncated.to_csv(outpath, index=False, header=False)

    removed = len(df) - len(df_truncated)
    print(f"{os.path.basename(path)}: truncated {removed} rows after index {index}")

def main():
    data_folder = os.path.join("simulation_data", SCENARIO)
    modes = []
    if PROCESS_COLLECTION:
        modes.append("collection")
    if PROCESS_EVAL:
        modes.append("eval")
    if not modes:
        print("Nothing to do: both PROCESS_COLLECTION and PROCESS_EVAL are False.")
        return

    for mode in modes:
        pattern = PATTERNS[mode]
        full_pattern = os.path.join(data_folder, pattern)
        files = glob.glob(full_pattern)
        if not files:
            print(f"No {mode} files found matching: {pattern}")
            continue
        print(f"\nProcessing {mode} files:")
        for path in files:
            if TRUNCATE_AFTER_INDEX is not None:
                truncate_csv_file(path, TRUNCATE_AFTER_INDEX, inplace=INPLACE)
            dedupe_csv_file(path, inplace=INPLACE)

if __name__ == "__main__":
    main()