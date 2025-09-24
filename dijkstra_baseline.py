import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def dijkstra_baseline(p_agents, p_tx, p_recv, Rcom=1.0, sigma=0.1, beta=0.99, r_del=1.0, c_pos=0.1):
    K = p_agents.shape[0]
    I = list(range(K))

    # --- Stage 1: Message retrieval ---
    G1 = nx.DiGraph()
    G1.add_node('s', pos=p_tx)
    G1.add_node('r', pos=p_recv)
    for k in I:
        G1.add_node(k, pos=p_agents[k])

    # Edge from sender to each agent
    for k in I:
        weight = np.linalg.norm(p_agents[k] - p_tx)
        G1.add_edge('s', k, weight=weight)

    # Edges between agents
    for k in I:
        for j in I:
            if k != j:
                weight = np.linalg.norm(p_agents[j] - p_agents[k])
                G1.add_edge(k, j, weight=weight)

    # Edges from agents to receiver
    for k in I:
        weight = np.linalg.norm(p_recv - p_agents[k])
        G1.add_edge(k, 'r', weight=weight)

    # Shortest path s->r
    path_stage1 = nx.dijkstra_path(G1, source='s', target='r', weight='weight')
    km = path_stage1[1]  # first agent that retrieves the message

    # --- Stage 2: Message delivery ---
    G2 = nx.DiGraph()
    for k in I:
        G2.add_node(k, pos=p_agents[k])
    G2.add_node('r', pos=p_recv)

    # Edge from km to all other agents and receiver
    pk = p_agents[km]
    s_km = ((np.linalg.norm(pk) - Rcom) / np.linalg.norm(pk)) * pk

    for j in I:
        if j != km:
            pj = p_agents[j]
            cos_theta = np.clip(np.dot(s_km, pj) / (np.linalg.norm(s_km)*np.linalg.norm(pj)), -1, 1)
            theta_kj = np.arccos(cos_theta)
            d_kj = np.sqrt(np.linalg.norm(s_km)**2 + np.linalg.norm(pj)**2 - 2*np.linalg.norm(s_km)*np.linalg.norm(pj)*np.cos(theta_kj))
            weight = (np.linalg.norm(pk)-Rcom) + (d_kj - Rcom)
            G2.add_edge(km, j, weight=weight)

    # Edge from km to receiver
    pj = p_recv
    cos_theta = np.clip(np.dot(s_km, pj) / (np.linalg.norm(s_km)*np.linalg.norm(pj)), -1, 1)
    theta_kr = np.arccos(cos_theta)
    d_kr = np.sqrt(np.linalg.norm(s_km)**2 + np.linalg.norm(pj)**2 - 2*np.linalg.norm(s_km)*np.linalg.norm(pj)*np.cos(theta_kr))
    weight = (np.linalg.norm(pk)-Rcom) + (d_kr - Rcom)
    G2.add_edge(km, 'r', weight=weight)

    # Edges between other agents and receiver
    for k in I:
        if k != km:
            for j in I:
                if j != km and j != k:
                    weight = np.linalg.norm(p_agents[j]-p_agents[k]) - Rcom
                    G2.add_edge(k,j,weight=weight)
            weight = np.linalg.norm(p_recv-p_agents[k]) - Rcom
            G2.add_edge(k,'r',weight=weight)

    # Shortest path from km to receiver
    path_stage2 = nx.dijkstra_path(G2, source=km, target='r', weight='weight')

    # --- Stage 3: Reward/cost computation ---
    D_nu2 = 0
    for i in range(len(path_stage2)-1):
        D_nu2 += G2.edges[path_stage2[i], path_stage2[i+1]]['weight']

    T_D = int(np.ceil(D_nu2 / sigma))
    V_piD = (beta**T_D)*r_del - c_pos * sigma**2 * (1 - beta**T_D)/(1-beta)

    return {
        'retrieving_agent': km,
        'path': path_stage2,
        'distance': D_nu2,
        'delivery_time': T_D,
        'value': V_piD
    }



def sample_scenario(K=5, Rcom=1.0, R=5.0, Ra=0.6, seed=None):
    rng = np.random.default_rng(seed)

    Rmin = K * Rcom
    Rmax = (K + 4) * Rcom
    dense = R <= (K + 1) * Rcom

    # Transmitting base at origin
    p_tx = np.array([0.0, 0.0])

    # Midpoint for agents initialization
    p0 = np.array([R/2, 0.0])

    # Agents positions around p0
    p_agents = rng.uniform(-Ra, Ra, size=(K, 2))
    p_agents = p0 + p_agents * rng.random((K, 1))
    #norms = np.linalg.norm(p_agents, axis=1, keepdims=True)
    #p_agents = p0 + p_agents / norms * rng.random((K, 1))

    # Receiver base
    p_recv = np.array([R, 0.0])

    return {
        'R': R,
        'Rmin': Rmin,
        'Rmax': Rmax,
        'dense': dense,
        'p_agents': p_agents,
        'p_recv': p_recv,
        'p0': p0,
        'p_tx': p_tx,
        'Rcom': Rcom,
        'Ra': Ra
    }


def plot_scenario(sample, savepath=None):
    R = sample['R']
    Rmin = sample['Rmin']
    Rmax = sample['Rmax']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
    p0 = sample['p0']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot agents
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', label='Agents')

    # Plot transmitting base at p_tx
    ax.scatter(*p_tx, c='green', marker='s', label='Transmitting base $p_{tx}$')
    circle_tx = plt.Circle(p_tx, Rcom, color='green', fill=False, linestyle='--', alpha=0.6)
    ax.add_artist(circle_tx)

    # Plot midpoint base p0
    ax.scatter(*p0, c='blue', marker='s', label='Midpoint $p_0$')
    circle_Ba = plt.Circle(p0, Ra, color='blue', fill=False, linestyle='--', alpha=0.6, label='$\mathbb{B}_a$')
    ax.add_artist(circle_Ba)

    # Gather all relevant points and radii
    points = np.vstack([p_agents, p0.reshape(1,2), p_tx.reshape(1,2), p_recv.reshape(1,2)])
    max_radius = Rcom  # circle radius around each base

    ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)


    # Add receiving base at (R,0)
    ax.scatter(*p_recv, c='green', marker='s', label='Receiver base')
    circle_base_rx = plt.Circle(p_recv, Rcom, color='green', fill=False, linestyle=':')
    ax.add_artist(circle_base_rx)

    # Annotations
    ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(K+1)*Rcom:.3f}", 
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()

def plot_scenario_with_path_colored(sample, dijkstra_result, savepath=None):
    R = sample['R']
    Rmin = sample['Rmin']
    Rmax = sample['Rmax']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
    p0 = sample['p0']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot agents
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', label='Agents')

    # Transmitting base
    ax.scatter(*p_tx, c='red', marker='s', label='Transmitting base $p_{tx}$')
    ax.add_artist(plt.Circle(p_tx, Rcom, color='red', fill=False, linestyle='--', alpha=0.6))

    # Midpoint base p0
    ax.scatter(*p0, c='blue', marker='s', label='Midpoint $p_0$')
    ax.add_artist(plt.Circle(p0, Ra, color='blue', fill=False, linestyle='--', alpha=0.6))

    # Receiver base
    ax.scatter(*p_recv, c='green', marker='s', label='Receiver base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color='green', fill=False, linestyle=':'))

    # Node positions
    node_positions = {'s': p_tx, 'r': p_recv}
    for i, pos in enumerate(p_agents):
        node_positions[i] = pos

    path = dijkstra_result['path']
    km = dijkstra_result['retrieving_agent']

    # Stage 1: Message retrieval arrows (agent -> transmitter -> agent)
    ax.annotate("", xy=node_positions['s'], xytext=node_positions[km],
                arrowprops=dict(arrowstyle="->", color='orange', lw=2))
    ax.annotate("", xy=node_positions[km], xytext=node_positions['s'],
                arrowprops=dict(arrowstyle="->", color='orange', lw=2))

    # Stage 2: Message delivery arrows along Dijkstra path (km -> ... -> r)
    for i in range(len(path)-1):
        start = node_positions[path[i]]
        end = node_positions[path[i+1]]
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color='red', lw=2))

    # Adjust plot limits
    ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)

    # Annotations
    ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(len(p_agents)+1)*Rcom:.3f}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Custom legend for arrows
    legend_elements = [
        Line2D([0], [0], color='orange', lw=2, label='Message retrieval'),
        Line2D([0], [0], color='red', lw=2, label='Message delivery')
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_elements)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()




if __name__ == '__main__':
    K = 5
    Rcom = 1.0
    R = (K+4)*Rcom
    Ra = 0.6*R

    sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, seed=42)
    result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom)    
    plot_scenario_with_path_colored(sample, result, savepath='Plots/scenario_with_path.png')
