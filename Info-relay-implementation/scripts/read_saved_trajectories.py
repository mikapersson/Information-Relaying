import pickle
import pandas as pd

def load_episodes(filepath):
    """
    Load episodes stored in nested dict format from a pickle file.
    
    Returns:
        episodes_dict: dict of dicts of dicts {episode_idx: {timestep: {column: value}}}
        df: Pandas DataFrame concatenating all episodes for easy analysis
    """
    # Load nested dict
    with open(filepath, "rb") as f:
        episodes_dict = pickle.load(f)

    # Convert nested dict to DataFrame
    all_dfs = []
    for ep_idx, timesteps in episodes_dict.items():
        ep_df = pd.DataFrame.from_dict(timesteps, orient="index")
        ep_df["episode_idx"] = ep_idx  # keep episode index column
        all_dfs.append(ep_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    return episodes_dict, full_df

if __name__ == "__main__":

    # Load from pickle file
    episodes_dict, full_df = load_episodes("/home/u099435/Info_relay_project/Information-Relaying/Info-relay-implementation/TEST_MAPPO_evaluation_results_K5_cpos0.5_cphi0.1_n10000_dir1_jam1.pkl")

    # Access nested dict
    #print(episodes_dict[1][0])  # episode 0, timestep 0
    #print(episodes_dict)

    # Look at the full DataFrame
    #print(full_df.head())
    print(full_df.columns)
    #print(episodes_dict[1])
    #print(full_df[full_df["episode_idx" == 1]]) 

    episode_1_df = full_df[full_df["episode_idx"] == 1] # how to access a specific episode - the first episode is 1, not 0

    # accessing specifc columns in the episode dataframe: 
    cols = ["agent0_x", "agent0_y", "agent0_phi"]
    agent0_data = episode_1_df[cols]
    print(agent0_data) 