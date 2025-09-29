import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


def compute_retrieval_point(pk, Rcom):
    norm_pk = np.linalg.norm(pk)
    if norm_pk > Rcom:
        return (Rcom / norm_pk) * pk
    else:
        return pk

def compute_projection(pj, ps_k, pr):
    v = pr - ps_k
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        return ps_k.copy()  # degenerate case
    alpha = np.dot(pj - ps_k, v) / v_norm_sq
    # Optionally clamp alpha to [0, 1] if you want projections only on the segment
    # alpha = np.clip(alpha, 0, 1)
    return ps_k + alpha * v

def compute_relay_point(pj, ps_k, pk, pr, j_idx, Rcom):
    # --- Compute unit direction u ---
    u = pr - ps_k
    norm_u = np.linalg.norm(u)
    if norm_u == 0:
        u = np.zeros_like(u)
    else:
        u = u / norm_u

    # --- Orthogonal projection of pj onto line ---
    alpha_j = np.dot(pj - ps_k, u)
    bar_pj = ps_k + alpha_j * u

    # --- Parameters ---
    b = np.linalg.norm(pj - bar_pj)
    D = np.linalg.norm(ps_k - pk)
    a = np.linalg.norm(bar_pj - ps_k) - (j_idx-1)*Rcom

    # --- Condition for direct projection ---
    cond = b <= D - np.linalg.norm(bar_pj - ps_k) - (j_idx-1)*Rcom
    if cond:
        return bar_pj

    # --- Fallback: solve for q_j ---
    n = (pj - bar_pj) / (b + 1e-12)  # unit normal from line to pj

    # Quadratic coefficients
    A = 4*(b**2 - D**2)
    B = 4*b*(D**2 + a**2 - b**2)
    C = (D**2 + a**2 - b**2)**2 - 4*D**2*a**2

    disc = B**2 - 4*A*C
    if disc < 0 or A == 0:
        return bar_pj  # fallback to projection

    t1 = (-B + np.sqrt(disc)) / (2*A)
    t2 = (-B - np.sqrt(disc)) / (2*A)

    # Candidates
    q1 = bar_pj + t1 * n
    q2 = bar_pj + t2 * n

    # Admissible root: same side as pj and original equation holds
    candidates = []
    for t, q in zip([t1, t2], [q1, q2]):
        if np.dot(q - bar_pj, pj - bar_pj) >= 0:
            # Check original (unsquared) equation
            lhs = np.abs(b - t)
            rhs = D + np.sqrt(a**2 + t**2)
            if np.isclose(lhs, rhs, atol=1e-6):
                candidates.append(q)
    if candidates:
        return candidates[0]
    else:
        return bar_pj

def dijkstra_baseline(p_agents, p_tx, p_recv, Rcom=1.0, sigma=0.1, beta=0.99, r_del=1.0, c_pos=0.1):
    K = p_agents.shape[0]
    I = list(range(K))
    best_path = None
    best_distance = np.inf
    best_k = None
    best_graph = None
    best_relay_points = None

    # Store all candidate paths and relay points for later agent movement computation
    candidate_paths = []
    candidate_relay_points = []
    candidate_graphs = []
    candidate_k = []

    for k in I:
        pk = p_agents[k]
        ps_k = compute_retrieval_point(pk, Rcom)
        pr = p_recv
        v = pr - ps_k
        v_norm_sq = np.dot(v, v)
        # Build graph G_k
        Gk = nx.DiGraph()
        node_map = {}  # maps agent idx to relay point
        Gk.add_node(k, pos=pk)  # retrieving agent
        Gk.add_node('s_k', pos=ps_k)
        Gk.add_node('r', pos=pr)
        # Add relay agents
        for j in I:
            if j == k:
                continue
            pj = p_agents[j]
            pmeet_j = compute_projection(pj, ps_k, pr)
            alpha = np.dot(pj - ps_k, pr - ps_k) / np.dot(pr - ps_k, pr - ps_k)
            # Option 1: Remove the alpha check entirely
            # (include all agents as relay candidates)
            # Or
            
            # Option 2: Use a tolerance
            tol = 0.2  # for example
            if alpha < -tol or alpha > 1+tol:
                continue  # skip agents whose projection is not on the segment
            relay_j = compute_relay_point(pj, ps_k, pk, pr, j_idx=(j+1), Rcom=Rcom)
            node_map[j] = relay_j
            Gk.add_node(j, pos=relay_j)
        # Edges
        Gk.add_edge(k, 's_k', weight=np.linalg.norm(ps_k - pk))
        Gk.add_edge('s_k', 'r', weight=max(0, np.linalg.norm(pr - ps_k) - Rcom))
        for j, relay_j in node_map.items():
            w = max(0, np.linalg.norm(relay_j - ps_k) - Rcom)
            Gk.add_edge('s_k', j, weight=w)
            Gk.add_edge(j, 'r', weight=max(0, np.linalg.norm(pr - relay_j) - Rcom))
        for j, relay_j in node_map.items():
            for l, relay_l in node_map.items():
                if j != l:
                    Gk.add_edge(j, l, weight=max(0, np.linalg.norm(relay_l - relay_j) - Rcom))
        # Run Dijkstra from k to r
        try:
            path = nx.dijkstra_path(Gk, source=k, target='r', weight='weight')
            total_dist = 0
            for i in range(len(path)-1):
                total_dist += Gk.edges[path[i], path[i+1]]['weight']
            candidate_paths.append((path, total_dist))
            candidate_relay_points.append(node_map)
            candidate_graphs.append(Gk)
            candidate_k.append(k)
            if total_dist < best_distance:
                best_distance = total_dist
                best_path = path
                best_k = k
                best_graph = Gk
                best_relay_points = node_map
        except nx.NetworkXNoPath:
            continue

    # Find relaying agents in optimal path (excluding retrieving agent and bases)
    relaying_agents = [n for n in best_path if isinstance(n, int) and n != best_k]
    k_D = len(relaying_agents)
    T_M = int(np.ceil(best_distance / sigma))
    T_D = T_M + k_D + 2

    # Compute per-agent distances and movement times
    D_k = np.zeros(K)
    t_start_k = np.zeros(K, dtype=int)
    t_stop_k = np.zeros(K, dtype=int)
    # Map agent index to relay point in path
    relay_points_in_path = []
    for n in best_path:
        if isinstance(n, int) and n != best_k:
            relay_points_in_path.append(best_relay_points[n])
    # For retrieving agent
    if k_D > 0:
        first_relay = relaying_agents[0]
        # Use retrieving agent's position and retrieval point
        D_k[best_k] = max(0, np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) - Rcom) \
            + max(0, np.linalg.norm(best_relay_points[first_relay] - best_graph.nodes['s_k']['pos']) - Rcom)
    else:
        D_k[best_k] = max(0, np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) - Rcom)
    # For relaying agents
    for idx, k in enumerate(relaying_agents):
        relay_point = best_relay_points[k]
        if idx < k_D - 1:
            next_relay = relaying_agents[idx+1]
            D_k[k] = np.linalg.norm(relay_point - p_agents[k]) \
                + max(0, np.linalg.norm(best_relay_points[next_relay] - relay_point) - Rcom)
        else:
            # Last relay agent
            D_k[k] = np.linalg.norm(relay_point - p_agents[k]) \
                + max(0, np.linalg.norm(p_recv - relay_point) - Rcom)
    # Movement times
    for idx, k in enumerate(relaying_agents):
        if idx == 0:
            D_kstar_k = max(0, np.linalg.norm(best_relay_points[k] - best_graph.nodes['s_k']['pos']) - Rcom) \
                + max(0, np.linalg.norm(best_relay_points[k] - best_graph.nodes['s_k']['pos']) - Rcom)
            t_start_k[k] = int(np.floor(max(0, D_kstar_k - np.linalg.norm(best_relay_points[k] - p_agents[k])) / sigma)) + idx
        else:
            prev_relay = relaying_agents[idx-1]
            D_prev = np.linalg.norm(best_relay_points[k] - best_relay_points[prev_relay])
            t_start_k[k] = int(np.floor(max(0, D_prev - np.linalg.norm(best_relay_points[k] - p_agents[k])) / sigma)) + idx
        t_stop_k[k] = t_start_k[k] + int(np.ceil(D_k[k] / sigma))
    # Retrieving agent
    t_stop_k[best_k] = int(np.ceil(D_k[best_k] / sigma))

    # Value computation
    V_piD = (beta**T_D)*r_del
    for t in range(T_D+1):
        for k in range(K):
            if t_start_k[k] <= t < t_stop_k[k]:
                V_piD -= c_pos * sigma**2 * (beta**t)

    return {
        'retrieving_agent': best_k,
        'path': best_path,
        'distance': best_distance,
        'delivery_time': T_D,
        'value': V_piD,
        'relay_points': best_relay_points,
        'graph': best_graph,
        'agent_distances': D_k,
        'agent_start_times': t_start_k,
        'agent_stop_times': t_stop_k,
        'relaying_agents': relaying_agents
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
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', alpha=0.3, label='Initial agent positions')

    # Plot transmitting base at p_tx
    ax.scatter(*p_tx, c='green', marker='s', label='Transmitting base $p_{tx}$')
    circle_tx = plt.Circle(p_tx, Rcom, color='green', fill=False, linestyle='--', alpha=0.6)
    ax.add_artist(circle_tx)

    # Plot midpoint base p0
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color='blue', fill=False, linestyle='--', alpha=0.6))

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
    receive_color = "red"
    deliver_color = "green"
    relay_color = "orange"

    # Plot initial agent positions (low opacity)
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', alpha=0.3)

    # Transmitting base
    ax.scatter(*p_tx, c=receive_color, marker='s', label='Transmitting base $p_{tx}$')
    ax.add_artist(plt.Circle(p_tx, Rcom, color=receive_color, fill=False, linestyle='--', alpha=0.6))

    # Midpoint base p0
    #ax.add_artist(plt.Circle(p0, Ra, color='blue', fill=False, linestyle='--', alpha=0.6))

    # Receiver base
    ax.scatter(*p_recv, c=deliver_color, marker='s', label='Receiver base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--'))

    # Node positions
    node_positions = {'s': p_tx, 'r': p_recv}
    for i, pos in enumerate(p_agents):
        node_positions[i] = pos

    # Add retrieval point for the retrieving agent
    km = dijkstra_result['retrieving_agent']
    ps_k = compute_retrieval_point(p_agents[km], sample['Rcom'])
    node_positions['s_k'] = ps_k

    # Plot the retrieval point in orange
    ax.scatter(ps_k[0], ps_k[1], c=relay_color, marker='o')

    # Plot dashed line between retrieval point and receiving base
    ax.plot([ps_k[0], p_recv[0]], [ps_k[1], p_recv[1]], 'o--', linewidth=1.5, alpha=0.4)  # label='Retrieval-to-receiver line'

    path = dijkstra_result['path']
    relay_points = dijkstra_result['relay_points']

    # Plot relay points (destinations) with full opacity
    for idx, relay in relay_points.items():
        ax.scatter(relay[0], relay[1], c=relay_color, alpha=1.0, marker='o', label='Relay points' if idx == list(relay_points.keys())[0] else "")
        # Add blue circle of radius Rcom around each relay point
        circle = plt.Circle(relay, Rcom, color=relay_color, fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)

    # Connect initial agent positions to their relay points with dashed lines
    for idx, relay in relay_points.items():
        agent_pos = p_agents[idx]
        ax.plot([agent_pos[0], relay[0]], [agent_pos[1], relay[1]], 'k--', alpha=0.7, linewidth=1)

    # Message retrieval arrow (agent -> transmitter)
    #ax.annotate("", xy=node_positions['s'], xytext=node_positions[km],
    #            arrowprops=dict(arrowstyle="->", color=receive_color, lw=2))

    # Message delivery arrows along Dijkstra path (km -> ... -> r)
    for i in range(len(path)-1):
        node_from = path[i]
        node_to = path[i+1]
        # Use relay points for agent nodes, otherwise use node_positions
        def get_plot_pos(node):
            if isinstance(node, int) and node in relay_points:
                return relay_points[node]
            else:
                return node_positions[node]
        start = get_plot_pos(node_from)
        end = get_plot_pos(node_to)
        # If next node is a relay agent, plot arrow to the edge of its Rcom circle
        if isinstance(node_to, int) and node_to in relay_points:
            direction = relay_points[node_to] - start
            norm = np.linalg.norm(direction)
            # Only plot arrow if distance > Rcom
            if norm > Rcom + 1e-8:
                direction = direction / norm
                end = relay_points[node_to] - direction * Rcom
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=2))
        elif node_to == 'r':
            # Last arrow: from last relay point to edge of receiver base Rcom circle
            direction = p_recv - start
            norm = np.linalg.norm(direction)
            if norm > Rcom + 1e-8:
                direction = direction / norm
                end = p_recv - direction * Rcom
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=2))
        else:
            # For other cases, plot arrow if distance > Rcom
            norm = np.linalg.norm(end - start)
            if norm > Rcom + 1e-8:
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=2))

    # Adjust plot limits
    ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)

    # Annotations
    #ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(len(p_agents)+1)*Rcom:.3f}",
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Custom legend for arrows and relay points
    legend_elements = [
        #Line2D([0], [0], color=receive_color, lw=2, label='Message retrieval'),
        Line2D([0], [0], color=deliver_color, lw=2, label='Message delivery path'),
        Line2D([0], [0], marker='o', color='w', label='Initial agent positions', markerfacecolor='blue', alpha=0.3)
        #Line2D([0], [0], marker='o', color='w', label='Relay point', markerfacecolor='orange', markersize=8),
        #Line2D([0], [0], linestyle='--', color='k', label='Agent to relay', alpha=0.7)
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + [le.get_label() for le in legend_elements])

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"Dijkstra Baseline ($K$: {K}; Time steps: {dijkstra_result['delivery_time']})")

    if savepath:
        plt.savefig(savepath, dpi=300)
    #plt.show()

def animate_dijkstra(sample, dijkstra_result, savepath='dijkstra_animation.mp4'):
    R = sample['R']
    Rmin = sample['Rmin']
    Rmax = sample['Rmax']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
    p0 = sample['p0']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    agent_start_times = dijkstra_result['agent_start_times']
    agent_stop_times = dijkstra_result['agent_stop_times']
    agent_distances = dijkstra_result['agent_distances']
    relay_points = dijkstra_result['relay_points']
    relaying_agents = dijkstra_result['relaying_agents']
    best_k = dijkstra_result['retrieving_agent']
    T_D = dijkstra_result['delivery_time']

    # Precompute agent trajectories (linear interpolation)
    agent_traj = []
    for k in range(len(p_agents)):
        if k == best_k:
            dest = compute_retrieval_point(p_agents[k], Rcom)
        elif k in relay_points:
            dest = relay_points[k]
        else:
            dest = p_agents[k]
        agent_traj.append((p_agents[k], dest))

    fig, ax = plt.subplots(figsize=(6, 6))
    receive_color = "red"
    deliver_color = "green"
    relay_color = "orange"

    def update(t):
        ax.clear()
        # Plot static elements
        ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', alpha=0.3, label='Initial agent positions')
        ax.scatter(*p_tx, c=receive_color, marker='s', label='Transmitting base $p_{tx}$')
        ax.add_artist(plt.Circle(p_tx, Rcom, color=receive_color, fill=False, linestyle='--', alpha=0.6))
        ax.scatter(*p_recv, c=deliver_color, marker='s', label='Receiver base')
        ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--'))
        ax.add_artist(plt.Circle(p0, 0.6*R, color='blue', fill=False, linestyle='--', alpha=0.6))

        # Plot relay points and their circles
        for idx, relay in relay_points.items():
            ax.scatter(relay[0], relay[1], c=relay_color, alpha=1.0, marker='o')
            circle = plt.Circle(relay, Rcom, color=relay_color, fill=False, linestyle='--', alpha=0.5)
            ax.add_artist(circle)

        # Plot retrieval point in orange
        km = dijkstra_result['retrieving_agent']
        ps_k = compute_retrieval_point(p_agents[km], Rcom)
        ax.scatter(ps_k[0], ps_k[1], c=relay_color, marker='o')

        # Plot dashed line between retrieval point and receiving base
        ax.plot([ps_k[0], p_recv[0]], [ps_k[1], p_recv[1]], 'o--', linewidth=1.5, alpha=0.4)

        # Plot agent positions at time t
        for k in range(len(p_agents)):
            t_start = agent_start_times[k]
            t_stop = agent_stop_times[k]
            start, end = agent_traj[k]
            if t < t_start:
                pos = start
            elif t >= t_stop:
                pos = end
            else:
                frac = (t - t_start) / max(1, t_stop - t_start)
                pos = start + frac * (end - start)
            ax.scatter(pos[0], pos[1], c='blue', s=80, alpha=1.0)

        # Plot relay arrows (only if agent has started moving)
        path = dijkstra_result['path']
        for i in range(len(path)-1):
            node_from = path[i]
            node_to = path[i+1]
            def get_plot_pos(node):
                if isinstance(node, int) and node in relay_points:
                    return relay_points[node]
                elif node == 's_k':
                    return ps_k
                elif node == 'r':
                    return p_recv
                else:
                    return p_agents[node]
            start = get_plot_pos(node_from)
            end = get_plot_pos(node_to)
            # If next node is a relay agent, plot arrow to the edge of its Rcom circle
            if isinstance(node_to, int) and node_to in relay_points:
                direction = relay_points[node_to] - start
                norm = np.linalg.norm(direction)
                if norm > Rcom + 1e-8:
                    direction = direction / norm
                    arrow_end = relay_points[node_to] - direction * Rcom
                    ax.annotate("", xy=arrow_end, xytext=start,
                                arrowprops=dict(arrowstyle="->", color=deliver_color, lw=2))
            elif node_to == 'r':
                direction = p_recv - start
                norm = np.linalg.norm(direction)
                if norm > Rcom + 1e-8:
                    direction = direction / norm
                    arrow_end = p_recv - direction * Rcom
                    ax.annotate("", xy=arrow_end, xytext=start,
                                arrowprops=dict(arrowstyle="->", color=deliver_color, lw=2))

        ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
        ax.set_ylim(-1.1*Ra, 1.1*Ra)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Dijkstra Baseline Animation (timestep {t}/{T_D})")

    ani = animation.FuncAnimation(fig, update, frames=range(dijkstra_result['delivery_time']+1), interval=300)
    writer = FFMpegWriter(fps=3, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(savepath, writer=writer, dpi=200)
    plt.close(fig)
    

def animate_dijkstra_minimal(sample, dijkstra_result):
    R = sample['R']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    agent_start_times = dijkstra_result['agent_start_times']
    agent_stop_times = dijkstra_result['agent_stop_times']
    relay_points = dijkstra_result['relay_points']
    best_k = dijkstra_result['retrieving_agent']
    T_D = dijkstra_result['delivery_time']

    # Precompute agent trajectories (linear interpolation)
    agent_traj = []
    for k in range(len(p_agents)):
        if k == best_k:
            dest = compute_retrieval_point(p_agents[k], Rcom)
        elif k in relay_points:
            dest = relay_points[k]
        else:
            dest = p_agents[k]
        agent_traj.append((p_agents[k], dest))

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(t):
        ax.clear()
        ax.scatter(p_agents[:, 0], p_agents[:, 1], c='blue', alpha=0.3)
        ax.scatter(*p_tx, c='red', marker='s')
        ax.scatter(*p_recv, c='green', marker='s')
        ax.add_artist(plt.Circle(p_tx, Rcom, color='red', fill=False, linestyle='--', alpha=0.6))
        ax.add_artist(plt.Circle(p_recv, Rcom, color='green', fill=False, linestyle='--'))
        # Plot relay points
        for relay in relay_points.values():
            ax.scatter(relay[0], relay[1], c='orange', marker='o')
        # Plot agent positions at time t
        for k in range(len(p_agents)):
            t_start = agent_start_times[k]
            t_stop = agent_stop_times[k]
            start, end = agent_traj[k]
            if t < t_start:
                pos = start
            elif t >= t_stop:
                pos = end
            else:
                frac = (t - t_start) / max(1, t_stop - t_start)
                pos = start + frac * (end - start)
            ax.scatter(pos[0], pos[1], c='blue', s=80, alpha=1.0)
        ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
        ax.set_ylim(-1.1*Ra, 1.1*Ra)
        ax.set_aspect('equal')
        ax.set_title(f"Timestep {t}/{T_D}")

    ani = animation.FuncAnimation(fig, update, frames=range(T_D+1), interval=300)
    ani.save('dijkstra_animation.gif', writer='pillow', dpi=120)
    plt.close(fig)
    


if __name__ == '__main__':
    K = 10
    Rcom = 1.0
    R = (K+4)*Rcom
    Ra = 0.6*R
    seed=5
    save_path = fr'Plots/dijkstra_baseline_K{K}_seed{seed}.png'

    sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, seed=seed)
    result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom)    
    plot_scenario_with_path_colored(sample, result, savepath=save_path)

    # Uncomment to generate animation
    #animate_dijkstra(sample, result, savepath='dijkstra_animation.mp4')
    animate_dijkstra_minimal(sample, result)
