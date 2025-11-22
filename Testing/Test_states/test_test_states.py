import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_test_states(
    csv_path, K=5, n_show=10, alpha=0.15,
    plot_agents=True,
    plot_jammer=True,
    plot_agent_arrows=True,
    plot_jammer_arrow=True
):
    df = pd.read_csv(csv_path)
    n_states = len(df)
    n_show = min(n_show, n_states)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Draw agent instantiation circle (all scenarios share the same p0 and Ra)
    row0 = df.iloc[0]
    p_tx = np.array([row0['p_tx_x'], row0['p_tx_y']])
    p_rx = np.array([row0['p_rx_x'], row0['p_rx_y']])
    Ra = row0['Ra']
    p0 = (p_tx + p_rx) / 2
    circle_agents = plt.Circle(p0, Ra, color='grey', fill=False, linestyle='-', linewidth=2, alpha=0.7, label='Agent instantiation circle')
    ax.add_patch(circle_agents)

    # Draw jammer capsule (rectangle + two half circles)
    R = row0['R']
    Rcom = row0['Rcom']
    height = 3 * Rcom
    radius = 1.5 * Rcom

    # Rectangle
    rect = plt.Rectangle((0, -height/2), R, height, edgecolor='purple', facecolor='none', lw=2, alpha=0.4, label='Jammer capsule')
    ax.add_patch(rect)

    # Left half circle
    circle_left = plt.Circle((0, 0), radius, edgecolor='purple', facecolor='none', lw=2, alpha=0.4)
    ax.add_patch(circle_left)

    # Right half circle
    circle_right = plt.Circle((R, 0), radius, edgecolor='purple', facecolor='none', lw=2, alpha=0.4)
    ax.add_patch(circle_right)

    arrow_length = Rcom
    for i in range(n_show):
        row = df.iloc[i]
        R = row['R']
        Rcom = row['Rcom']
        Ra = row['Ra']
        p_tx = np.array([row['p_tx_x'], row['p_tx_y']])
        p_rx = np.array([row['p_rx_x'], row['p_rx_y']])
        jammer_pos = np.array([row['jammer_x'], row['jammer_y']])
        jammer_disp = np.array([row['jammer_dx'], row['jammer_dy']])

        agent_pos = np.array([[row[f'agent{k}_x'], row[f'agent{k}_y']] for k in range(1,K+1)])
        agent_phi = np.array([row[f'agent{k}_phi'] for k in range(1,K+1)])

        # Plot bases
        ax.scatter(*p_tx, c='blue', marker='s', s=100, alpha=alpha)
        ax.scatter(*p_rx, c='green', marker='s', s=100, alpha=0.1)

        # Plot agents
        if plot_agents:
            ax.scatter(agent_pos[:,0], agent_pos[:,1], c='orange', s=80, alpha=alpha)

        # Plot agent orientations as arrows
        if plot_agent_arrows:
            for k in range(K):
                ax.arrow(agent_pos[k,0], agent_pos[k,1],
                         np.sqrt(arrow_length)*np.cos(agent_phi[k]), np.sqrt(arrow_length)*np.sin(agent_phi[k]),
                         head_width=0.1*Rcom, head_length=0.1*Rcom, fc='orange', ec='orange', alpha=alpha)
        # Plot jammer
        if plot_jammer:
            ax.scatter(*jammer_pos, c='red', marker='x', s=100, alpha=alpha)

        # Plot jammer displacement as arrow
        if plot_jammer_arrow:
            ax.arrow(jammer_pos[0], jammer_pos[1], jammer_disp[0], jammer_disp[1],
                     head_width=0.2*arrow_length, head_length=0.3*arrow_length, fc='red', ec='red', alpha=alpha)
        # Plot Rcom circles for bases
        circle_tx = plt.Circle(p_tx, Rcom, color='blue', fill=False, linestyle='--', alpha=0.3*alpha)
        circle_rx = plt.Circle(p_rx, Rcom, color='green', fill=False, linestyle='--', alpha=0.3*alpha)
        ax.add_patch(circle_tx)
        ax.add_patch(circle_rx)

    ax.set_aspect('equal')
    ax.set_title(f"Overlay of {n_show} scenarios w. K={K} agents")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    K = 1
    n_total = 10000
    n_show = 300
    alpha = 0.4

    # Example usage:
    csv_path = f"Testing/Test_states/test_states_K{K}_n{n_total}.csv"
    plot_test_states(
        csv_path, K=K, n_show=n_show, alpha=alpha,
        plot_agents=False,
        plot_jammer=False,
        plot_agent_arrows=False,
        plot_jammer_arrow=True
    )