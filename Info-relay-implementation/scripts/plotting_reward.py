import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

mpl.rcParams['font.size'] = 14  # global font size

# ====== Configuration ======
scenarios = [
    "1_agent_one_way",
    "2_agent_one_way",
    "3_agent_one_way",
    "4_agent_one_way",
    "5_agent_one_way"
]

mode = "eval"
save_fn = f"plots_new/all_one_way_{mode}_reward_plot.png"

show_min_max = False
show_ci95 = True
show_ci90 = False
zoom_in = False
v_line = False

UPPER_YLIM = 100
LOWER_YLIM = None

X_RANGE = (None, None)

labels = ["1 agent", "2 agents", "3 agents", "4 agents", "5 agents"] 
# ===========================

# --- Main figure setup ---
fig, ax = plt.subplots(figsize=(6.4, 4.5))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
interval_colors = ['lightblue', 'navajowhite', 'palegreen', 'lightcoral', 'thistle']
linestyles = ['-']#['-', '--', '-.', ':', '-']
linewidth = 0.5 if mode == "collection" else 1

for i, scenario in enumerate(scenarios):
    folder = f"simulation_data/{scenario}"
    prefix = f"{mode}_agent_reward_episode_reward"

    main_file = os.path.join(folder, f"{prefix}_mean.csv")
    if not os.path.exists(main_file):
        print(f"Skipping scenario {scenario}: missing {main_file}")
        continue

    df_main = pd.read_csv(main_file, header=None, names=['x', 'y'])

    if X_RANGE[0] is not None or X_RANGE[1] is not None:
        df_main = df_main[df_main['x'].between(
            X_RANGE[0] if X_RANGE[0] is not None else df_main['x'].min(),
            X_RANGE[1] if X_RANGE[1] is not None else df_main['x'].max()
        )]

    # Shaded areas (optional)
    if show_min_max:
        df_lower = pd.read_csv(os.path.join(folder, f"{prefix}_min.csv"), header=None, names=['x', 'y'])
        df_upper = pd.read_csv(os.path.join(folder, f"{prefix}_max.csv"), header=None, names=['x', 'y'])
        if X_RANGE[0] is not None or X_RANGE[1] is not None:
            df_lower = df_lower[df_lower['x'].between(X_RANGE[0] or df_lower['x'].min(), X_RANGE[1] or df_lower['x'].max())]
            df_upper = df_upper[df_upper['x'].between(X_RANGE[0] or df_upper['x'].min(), X_RANGE[1] or df_upper['x'].max())]
        ax.fill_between(df_main['x'], df_lower['y'], df_upper['y'], color=interval_colors[i % len(colors)], alpha=0.5)

    if show_ci95:
        df_ci95_low = pd.read_csv(os.path.join(folder, f"{prefix}_ci95_low.csv"), header=None, names=['x', 'y'])
        df_ci95_high = pd.read_csv(os.path.join(folder, f"{prefix}_ci95_high.csv"), header=None, names=['x', 'y'])
        if X_RANGE[0] is not None or X_RANGE[1] is not None:
            df_ci95_low = df_ci95_low[df_ci95_low['x'].between(X_RANGE[0] or df_ci95_low['x'].min(), X_RANGE[1] or df_ci95_low['x'].max())]
            df_ci95_high = df_ci95_high[df_ci95_high['x'].between(X_RANGE[0] or df_ci95_high['x'].min(), X_RANGE[1] or df_ci95_high['x'].max())]
        ax.fill_between(df_main['x'], df_ci95_low['y'], df_ci95_high['y'], color=interval_colors[i % len(colors)], alpha=1)

    if show_ci90:
        df_ci90_low = pd.read_csv(os.path.join(folder, f"{prefix}_ci90_low.csv"), header=None, names=['x', 'y'])
        df_ci90_high = pd.read_csv(os.path.join(folder, f"{prefix}_ci90_high.csv"), header=None, names=['x', 'y'])
        if X_RANGE[0] is not None or X_RANGE[1] is not None:
            df_ci90_low = df_ci90_low[df_ci90_low['x'].between(X_RANGE[0] or df_ci90_low['x'].min(), X_RANGE[1] or df_ci90_low['x'].max())]
            df_ci90_high = df_ci90_high[df_ci90_high['x'].between(X_RANGE[0] or df_ci90_high['x'].min(), X_RANGE[1] or df_ci90_high['x'].max())]
        ax.fill_between(df_main['x'], df_ci90_low['y'], df_ci90_high['y'], color=interval_colors[i % len(colors)], alpha=0.3)

    # Plot mean line
    if labels is not None:
        ax.plot(df_main['x'], df_main['y'],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
            linewidth=linewidth,
            label=labels[i])
    else:
        label = scenario.replace("_", " ").capitalize()
        ax.plot(df_main['x'], df_main['y'],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
            linewidth=linewidth,
            label=label)

# Optional vertical line
if v_line:
    x_v = 250
    ax.axvline(x=x_v, color='red', linestyle='--', linewidth=1, label='Increased difficulty')

# Y-axis limits
if UPPER_YLIM is not None or LOWER_YLIM is not None:
    bottom, top = ax.get_ylim()
    ax.set_ylim(LOWER_YLIM if LOWER_YLIM is not None else bottom,
                UPPER_YLIM if UPPER_YLIM is not None else top)

ax.set_xlabel('Training Iteration')
ax.set_ylabel('ACER')
ax.grid(True)
ax.legend()#loc='lower right')
plt.tight_layout()
plt.savefig(save_fn, format='png')
plt.show()