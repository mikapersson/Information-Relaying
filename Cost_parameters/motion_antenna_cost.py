import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import numpy as np

# -------------------------
# Conservative (Single Agent)
# -------------------------
def compute_v_sharp(R, R_com, sigma, beta, c_delta_p, cphi):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    V = (beta ** t_max) - c_delta_p * sigma ** 2 * ((1 - beta ** (t_max + 1)) / (1 - beta))
    V -= cphi * (np.pi / 8)**2 * (beta ** (t_max - 7)) * (1 - beta ** 8) / (1 - beta)  # Antenna cost term
    return V

# -------------------------
# Less Conservative (All Agents)
# -------------------------
def compute_value_chain_policy(K, R_com, sigma, beta, c_delta_p, cphi):
    R = (K + 4) * R_com
    D1 = 1.1 * R + 2 * R_com
    t1_2 = int(np.ceil(D1 / sigma))
    t_max = t1_2 + K

    t_k_1 = np.zeros(K, dtype=int)
    t_k_2 = np.zeros(K, dtype=int)
    for k in range(1, K + 1):
        if k == 1:
            t_k_1[k - 1] = 0
            t_k_2[k - 1] = t1_2
        else:
            Dk = 1.1 * R - (3 + k) * R_com
            start_time = int(np.floor((D1 - Dk) / sigma)) + (k - 2)
            stop_time = t_max - (K - k + 1)
            t_k_1[k - 1] = start_time
            t_k_2[k - 1] = stop_time

    value = beta ** t_max
    motion_cost = 0.0
    antenna_cost = 0.0
    for n in range(t_max + 1):
        for k in range(K):
            if t_k_1[k] <= n <= t_k_2[k]:
                motion_cost += beta ** n
                if t_k_2[k]-7 <= n:
                    antenna_cost += beta ** n

    value -= c_delta_p * sigma ** 2 * motion_cost
    value -= cphi * (np.pi / 8)**2 * antenna_cost  # Antenna cost term
    return value

# -------------------------
# Comparison and 3D Plot
# -------------------------
def compare_value_functions_3d():
    # Fixed parameters
    K = 5
    R_com = 1
    R = (K + 4) * R_com
    sigma = 0.1
    beta = 0.99

    # Varying parameters
    motion_max_cost = 0.3
    antenna_max_cost = 0.1
    N = 50
    motion_cost_range = np.linspace(0.0, motion_max_cost, N)  # Range of motion cost (c_delta_p)
    antenna_cost_range = np.linspace(0.0, antenna_max_cost, N)  # Range of antenna cost (c_phi)

    # Initialize grid for 3D plot
    motion_cost_vals, antenna_cost_vals = np.meshgrid(motion_cost_range, antenna_cost_range)
    V_single_agent = np.zeros_like(motion_cost_vals)
    V_all_agents = np.zeros_like(motion_cost_vals)
    best_policy = np.zeros_like(motion_cost_vals, dtype=int)  # 0 = Passive, 1 = Single Agent, 2 = All Agents

    for i in range(len(motion_cost_range)):
        for j in range(len(antenna_cost_range)):
            c_delta_p = motion_cost_range[i]
            cphi = antenna_cost_range[j]

            # Compute values for both policies
            V_single = compute_v_sharp(R, R_com, sigma, beta, c_delta_p, cphi)
            V_all = compute_value_chain_policy(K, R_com, sigma, beta, c_delta_p, cphi)

            # Store the values
            V_single_agent[i, j] = V_single
            V_all_agents[i, j] = V_all

            # Determine the best policy
            if V_single >= V_all and V_single >= 0:
                best_policy[i, j] = 1  # Single Agent
            elif V_all > V_single and V_all > 0:
                best_policy[i, j] = 2  # All Agents
            else:
                best_policy[i, j] = 0  # Passive

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Custom color map for best_policy values: 0 = red, 1 = blue, 2 = green
    custom_cmap = ListedColormap(['r', 'b', 'g'])

    # Color the 2D plane according to the best policy (0: Passive, 1: Single Agent, 2: All Agents)
    c = ax.pcolormesh(motion_cost_vals, antenna_cost_vals, best_policy.T, cmap=custom_cmap, alpha=0.8)

    # Set the axis labels
    ax.set_xlabel('Motion Cost ($c_{\delta p}$)')
    ax.set_ylabel('Antenna Cost ($c_{\phi}$)')
    ax.set_title('Best Policy')

    # Manually add the legend for the policy colors
    proxy1 = mlines.Line2D([0], [0], color='b', lw=4, label='Single Agent')
    proxy2 = mlines.Line2D([0], [0], color='g', lw=4, label='All Agents')
    proxy3 = mlines.Line2D([0], [0], color='r', lw=4, label='Passive')


    """
    proxy4 = mlines.Line2D([0], [0], color='k', lw=2,
                           marker=r'$\rightarrow$', markersize=12,
                           linestyle='None', label='Gradient (All Agents)')
    # Add gradient of all-agent value function wrt cost parameters
    x_gradients = [0.05, 0.1, 0.15]
    y_gradients = [0.038, 0.024, 0.012]
    grad_motion = np.zeros_like(x_gradients)
    grad_antenna = np.zeros_like(y_gradients)
    for i in range(len(x_gradients)):
        c_delta_p = x_gradients[i]
        cphi = y_gradients[i]

        # Recompute with access to internal sums
        R = (K + 4) * R_com
        D1 = 1.1 * R + 2 * R_com
        t1_2 = int(np.ceil(D1 / sigma))
        t_max = t1_2 + K

        t_k_1 = np.zeros(K, dtype=int)
        t_k_2 = np.zeros(K, dtype=int)
        for k in range(1, K + 1):
            if k == 1:
                t_k_1[k - 1] = 0
                t_k_2[k - 1] = t1_2
            else:
                Dk = 1.1 * R - (3 + k) * R_com
                start_time = int(np.floor((D1 - Dk) / sigma)) + (k - 2)
                stop_time = t_max - (K - k + 1)
                t_k_1[k - 1] = start_time
                t_k_2[k - 1] = stop_time

        motion_cost = 0.0
        antenna_cost = 0.0
        for n in range(t_max + 1):
            for k in range(K):
                if t_k_1[k] <= n <= t_k_2[k]:
                    motion_cost += beta ** n
                    if t_k_2[k]-7 <= n:
                        antenna_cost += beta ** n

        # Gradients: derivative of value wrt each cost parameter
        grad_motion[i] = -sigma**2 * motion_cost
        grad_antenna[i] = -(np.pi/8)**2 * antenna_cost

    ax.quiver(x_gradients, y_gradients, grad_motion, grad_antenna, color="k", scale=20, width=0.01, zorder=3)
    """

    ax.legend(handles=[proxy1, proxy2, proxy3], loc='upper left')

    # Adjust layout to make sure everything fits without overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

    """
    # -------------------------
    # Compute gradients wrt cost parameters
    # -------------------------
    grad_motion = np.zeros_like(V_all_agents)
    grad_antenna = np.zeros_like(V_all_agents)

    for i in range(len(motion_cost_range)):
        for j in range(len(antenna_cost_range)):
            c_delta_p = motion_cost_range[i]
            cphi = antenna_cost_range[j]

            # Recompute with access to internal sums
            R = (K + 4) * R_com
            D1 = 1.1 * R + 2 * R_com
            t1_2 = int(np.ceil(D1 / sigma))
            t_max = t1_2 + K

            t_k_1 = np.zeros(K, dtype=int)
            t_k_2 = np.zeros(K, dtype=int)
            for k in range(1, K + 1):
                if k == 1:
                    t_k_1[k - 1] = 0
                    t_k_2[k - 1] = t1_2
                else:
                    Dk = 1.1 * R - (3 + k) * R_com
                    start_time = int(np.floor((D1 - Dk) / sigma)) + (k - 2)
                    stop_time = t_max - (K - k + 1)
                    t_k_1[k - 1] = start_time
                    t_k_2[k - 1] = stop_time

            motion_cost = 0.0
            antenna_cost = 0.0
            for n in range(t_max + 1):
                for k in range(K):
                    if t_k_1[k] <= n <= t_k_2[k]:
                        motion_cost += beta ** n
                        if t_k_2[k]-7 <= n:
                            antenna_cost += beta ** n

            # Gradients: derivative of value wrt each cost parameter
            grad_motion[i, j] = -sigma**2 * motion_cost
            grad_antenna[i, j] = -(np.pi/8)**2 * antenna_cost

    # -------------------------
    # Detect blue-green boundary
    # -------------------------
    border_points = []
    for i in range(1, len(motion_cost_range)-1):
        for j in range(1, len(antenna_cost_range)-1):
            if best_policy[i, j] in (1, 2):
                # Look at 4-neighborhood
                neighbors = [
                    best_policy[i+1, j],
                    best_policy[i-1, j],
                    best_policy[i, j+1],
                    best_policy[i, j-1],
                ]
                if 1 in neighbors and 2 in neighbors:
                    border_points.append((i, j))

    # -------------------------
    # Plot gradients only at border
    # -------------------------
    Xb, Yb, Ub, Vb = [], [], [], []
    for (i, j) in border_points:
        Xb.append(motion_cost_vals[j, j])
        Yb.append(antenna_cost_vals[j, i])
        Ub.append(grad_motion[j, i])
        Vb.append(grad_antenna[j, i])
    """


if __name__ == "__main__":
    compare_value_functions_3d()