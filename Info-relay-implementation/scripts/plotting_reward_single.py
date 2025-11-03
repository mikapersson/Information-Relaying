import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['font.size'] = 14  # global font size

# ====== Configuration ======
scenario = "1_agent_one_way_larger_distances"
mode     = "eval"

folder   = f"/home/u087303/relay/Information-Relaying/Info-relay-implementation/outputs/2025-11-02/10-45-28/mappo_info_relay_mlp__6c5871ef_25_11_02-10_45_29/mappo_info_relay_mlp__6c5871ef_25_11_02-10_45_29/scalars"
save_fn  = f"plots_new/{scenario}_{mode}_reward_plot.pdf"

show_min_max = True
show_ci95    = True
show_ci90    = False
zoom_in = False
v_line = False

UPPER_YLIM = 5 # 100
LOWER_YLIM = None # 0

# Select only a part of the data to plot (set to None to disable)
X_RANGE = (None, None) 
# ===========================

prefix     = f"{mode}_agent_reward_episode_reward"
main_file  = os.path.join(folder, f"{prefix}_mean.csv")
df_main    = pd.read_csv(main_file, header=None, names=['x', 'y'])

# Slice data if X_RANGE is set
if X_RANGE[0] is not None or X_RANGE[1] is not None:
    df_main = df_main[df_main['x'].between(
        X_RANGE[0] if X_RANGE[0] is not None else df_main['x'].min(),
        X_RANGE[1] if X_RANGE[1] is not None else df_main['x'].max()
    )]

max_ep = df_main['x'].max()


# read bounds if needed
if show_min_max:
    df_lower = pd.read_csv(os.path.join(folder, f"{prefix}_min.csv"), header=None, names=['x','y'])
    df_upper = pd.read_csv(os.path.join(folder, f"{prefix}_max.csv"), header=None, names=['x','y'])
    if X_RANGE[0] is not None or X_RANGE[1] is not None:
        df_lower = df_lower[df_lower['x'].between(X_RANGE[0] or df_lower['x'].min(), X_RANGE[1] or df_lower['x'].max())]
        df_upper = df_upper[df_upper['x'].between(X_RANGE[0] or df_upper['x'].min(), X_RANGE[1] or df_upper['x'].max())]

if show_ci95:
    df_ci95_low  = pd.read_csv(os.path.join(folder, f"{prefix}_ci95_low.csv"), header=None, names=['x','y'])
    df_ci95_high = pd.read_csv(os.path.join(folder, f"{prefix}_ci95_high.csv"), header=None, names=['x','y'])
    if X_RANGE[0] is not None or X_RANGE[1] is not None:
        df_ci95_low = df_ci95_low[df_ci95_low['x'].between(X_RANGE[0] or df_ci95_low['x'].min(), X_RANGE[1] or df_ci95_low['x'].max())]
        df_ci95_high = df_ci95_high[df_ci95_high['x'].between(X_RANGE[0] or df_ci95_high['x'].min(), X_RANGE[1] or df_ci95_high['x'].max())]

if show_ci90:
    df_ci90_low  = pd.read_csv(os.path.join(folder, f"{prefix}_ci90_low.csv"), header=None, names=['x','y'])
    df_ci90_high = pd.read_csv(os.path.join(folder, f"{prefix}_ci90_high.csv"), header=None, names=['x','y'])
    if X_RANGE[0] is not None or X_RANGE[1] is not None:
        df_ci90_low = df_ci90_low[df_ci90_low['x'].between(X_RANGE[0] or df_ci90_low['x'].min(), X_RANGE[1] or df_ci90_low['x'].max())]
        df_ci90_high = df_ci90_high[df_ci90_high['x'].between(X_RANGE[0] or df_ci90_high['x'].min(), X_RANGE[1] or df_ci90_high['x'].max())]

# --- Main figure ---
if mode == "collection":
    linewidth = 0.5
elif mode == "eval":
    linewidth = 1

fig, ax = plt.subplots(figsize=(6.4, 4.5))
ax.plot(df_main['x'], df_main['y'],
        marker='.', markersize=0.5, linewidth=linewidth, label='Mean Reward')

if show_min_max:
    ax.fill_between(df_main['x'],
                    df_lower['y'], df_upper['y'],
                    color='skyblue', alpha=0.3,
                    label='Min-Max Range')
if show_ci95:
    ax.fill_between(df_main['x'],
                    df_ci95_low['y'], df_ci95_high['y'],
                    color='tab:orange', alpha=0.5,
                    label='95% CI')
if show_ci90:
    ax.fill_between(df_main['x'],
                    df_ci90_low['y'], df_ci90_high['y'],
                    color='green', alpha=0.3,
                    label='90% CI')
    
if v_line:
    x_v = 2000  # or whatever x‑value you want
    ax.axvline(x=x_v, color='red', linestyle='--', linewidth=1, label=f'Increased difficulity')

if UPPER_YLIM is not None:
    # keep the current bottom, set top to UPPER_YLIM
    bottom, _ = ax.get_ylim()
    ax.set_ylim(bottom, UPPER_YLIM)

if LOWER_YLIM is not None:
    # keep the current bottom, set top to UPPER_YLIM
    _, upper = ax.get_ylim()
    ax.set_ylim(LOWER_YLIM, upper)



ax.set_xlabel('Training Iteration')
ax.set_ylabel('Average Cumulative Reward')
ax.grid(True)
ax.legend(loc='lower right')

if zoom_in:
    axins = inset_axes(ax,
                       width="30%",  # width of inset
                       height="20%",  # height of inset
                       loc='center')

    # Plot same shaded regions in the inset
    axins.plot(df_main['x'], df_main['y'],
               marker='.', markersize=0.5, linewidth=0.5)

    if show_min_max:
        axins.fill_between(df_main['x'],
                           df_lower['y'], df_upper['y'],
                           color='skyblue', alpha=0.3)
    if show_ci95:
        axins.fill_between(df_main['x'],
                           df_ci95_low['y'], df_ci95_high['y'],
                           color='orange', alpha=0.3)
    if show_ci90:
        axins.fill_between(df_main['x'],
                           df_ci90_low['y'], df_ci90_high['y'],
                           color='green', alpha=0.3)

    # zoom limits
    x0, x1 = max_ep - 250, max_ep
    axins.set_xlim(x0, x1)

    # optionally zoom y to data in that range:
    #y_slice = df_main['y'][df_main['x'].between(x0, x1)]
    #axins.set_ylim(y_slice.min(), y_slice.max())

    axins.set_ylim(90,100)

    # draw a box and connector lines
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.savefig(save_fn, format='pdf')
plt.show()