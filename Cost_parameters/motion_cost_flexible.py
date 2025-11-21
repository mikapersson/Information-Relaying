import numpy as np
import matplotlib.pyplot as plt

def compute_value_chain_policy(K, k_active, R_com, sigma, beta, c_delta_p, R=None):
    """
    Compute value function for using k_active agents (1 <= k_active <= K).
    """
    if R is None:
        R = (K + 4) * R_com

    D_k = np.zeros(k_active)
    t_k_1 = np.zeros(k_active, dtype=int)
    t_k_2 = np.zeros(k_active, dtype=int)

    # Distance each active agent must travel
    for k in range(1, k_active + 1):
        if k == 1:
            D_k[0] = 1.1 * R + (K - k_active + 2) * R_com
        else:
            D_k[k - 1] = 0.1 * R + (k_active - k + 1) * R_com

    D1 = D_k[0]
    t1_2 = int(np.ceil(D1 / sigma))
    t_max = t1_2 + k_active
    t_k_2[0] = t1_2
    t_k_1[0] = 0

    for k in range(2, k_active + 1):
        start = int(np.floor((D1 - D_k[k - 1]) / sigma)) + (k - 2)
        stop = t_max - (K - (k_active - k + 1))
        t_k_1[k - 1] = start
        t_k_2[k - 1] = stop

    # Value function
    value = (beta ** t_max)  # reward received at t_max
    motion_cost = 0.0

    for t in range(t_max + 1):
        for k in range(k_active):
            if t_k_1[k] <= t <= t_k_2[k]:
                motion_cost += beta ** t

    value -= c_delta_p * sigma ** 2 * motion_cost
    return value

# -------------------------------
# Comparison of all k_active
# -------------------------------
def compare_value_functions():
    # Parameters
    K = 5
    R_com = 1.0
    sigma = 0.1
    beta = 0.99
    c_values = np.linspace(0.0, 0.3, 200)

    # Storage
    value_dict = {k_act: [] for k_act in range(1, K+1)}

    for c in c_values:
        for k_act in range(1, K+1):
            val = compute_value_chain_policy(K, k_act, R_com, sigma, beta, c)
            value_dict[k_act].append(val)

    # Convert to arrays for plotting
    for k in value_dict:
        value_dict[k] = np.array(value_dict[k])

    # Plotting
    fig, (ax) = plt.subplots(figsize=(10, 12), sharex=True)

    # -- Top: All Value Functions
    for k_act in range(1, K+1):
        label = rf"$V_{{{k_act}}}$"
        ax.plot(c_values, value_dict[k_act], label=label)
        
      
    # Add x-axis arrow
    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.annotate("", xy=(x_max, 0), xytext=(x_min, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    # Add y-axis arrow
    ax.annotate("", xy=(0, y_max), xytext=(0, y_min),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
    
    # Remove default axes lines (optional, for cleaner look)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

        # === Shade under the max value line using the same color ===

    # Stack all value functions into a 2D array (shape: [K, len(c_values)])
    all_values = np.stack([value_dict[k] for k in range(1, K+1)])  # 1-indexed
    # Add the passive policy (y = 0 line) as an additional value function
    passive_values = np.zeros_like(c_values)
    all_values = np.vstack([all_values, passive_values])  # Now shape = [K+1, len(c_values)]

    # Get the index (0-based) of the highest value at each c_value
    max_indices = np.argmax(all_values, axis=0)

    # Use the same colormap and define passive policy color
    color_map = plt.get_cmap('tab10')
    line_colors = {k: color_map(k - 1) for k in range(1, K + 1)}
    passive_color = 'black'

    # Passive policy
    label = fr"$V_{{\mathrm{{passive}}}}$"
    ax.plot(c_values, passive_values, label=label, color=passive_color, linestyle='--')


    # Shade regions under the max value line (including passive policy)
    start_idx = 0
    for i in range(1, len(c_values)):
        if max_indices[i] != max_indices[start_idx]:
            idx_top = max_indices[start_idx]
            if idx_top < K:  # Active agents (1 to K)
                k_top = idx_top + 1
                color = line_colors[k_top]
            else:  # Passive policy
                color = passive_color

            ax.fill_between(
                c_values[start_idx:i],
                all_values[idx_top, start_idx:i],
                color=color,
                alpha=0.2
            )
            start_idx = i

    # Fill the last region
    idx_top = max_indices[start_idx]
    if idx_top < K:
        k_top = idx_top + 1
        color = line_colors[k_top]
    else:
        color = passive_color

    ax.fill_between(
        c_values[start_idx:],
        all_values[idx_top, start_idx:],
        color=color,
        alpha=0.2
    )






    ax.set_title(fr"Value functions for varying active agents ($K$={K}, $R_{{\mathrm{{com}}}}$ = {R_com}, $\sigma$={sigma}, $\beta$={beta})")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_value_functions()
