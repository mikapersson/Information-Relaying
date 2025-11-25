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
    
    return episodes_dict

def convert_dicts_to_df(episodes_dict):
    # Convert nested dict to DataFrame
    all_dfs = []
    for ep_idx, timesteps in episodes_dict.items():
        ep_df = pd.DataFrame.from_dict(timesteps, orient="index")
        ep_df["episode_idx"] = ep_idx  # keep episode index column
        all_dfs.append(ep_df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    
    return full_df

def select_episode(data, episode_idx):
    """
    load_episodes() has to have been called before this function
    Select a single episode from either:
      - episodes_dict  → returns episode_dict
      - full_df        → returns episode_df (DataFrame)

    Args:
        data: episodes_dict (dict) OR full_df (pd.DataFrame)
        episode_idx (int): which episode to extract

    Returns:
        episode_dict OR episode_df depending on input type
    """

    # ---------- Case 1: Nested dict input ----------
    if isinstance(data, dict):
        episodes_dict = data

        if episode_idx not in episodes_dict:
            raise KeyError(
                f"Episode {episode_idx} not found in episodes_dict. "
                f"Available episodes: {list(episodes_dict.keys())}"
            )

        return episodes_dict[episode_idx]

    # ---------- Case 2: DataFrame input ----------
    elif isinstance(data, pd.DataFrame):
        full_df = data

        if "episode_idx" not in full_df.columns:
            raise KeyError("DataFrame does not contain 'episode_idx' column!")

        episode_df = full_df[full_df["episode_idx"] == episode_idx].copy()

        if len(episode_df) == 0:
            raise KeyError(
                f"Episode {episode_idx} not found in DataFrame. "
                f"Available episodes: "
                f"{sorted(full_df['episode_idx'].unique().tolist())}"
            )

        return episode_df


if __name__ == "__main__":

    # Load from pickle file
    episodes_dict = load_episodes("/home/u099435/Info_relay_project/Information-Relaying/Info-relay-implementation/MAPPO_evaluation_results_K5_cpos0.5_cphi0.1_n10000_dir0_jam1.pkl")
    full_df = convert_dicts_to_df(episodes_dict)
    
    # Access nested dict
    print(episodes_dict[1][0])  # episode 1, timestep 0
    #print(episodes_dict)

    #print(full_df.head())
    #print(full_df.columns)
    #print(episodes_dict[1])
    #print(full_df[full_df["episode_idx" == 1]]) 

    #episode_1_df = full_df[full_df["episode_idx"] == 1] 
    episode_1_df = select_episode(full_df, 1) # how to access a specific episode - the first episode is 1, not 0

    # accessing specifc columns/values in the episode dataframe: 
    cols = ["agent0_x", "agent0_y", "agent0_phi"]
    agent0_data = episode_1_df[cols]
    print(agent0_data) 