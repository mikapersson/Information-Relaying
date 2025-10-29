import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


def compute_retrieval_point(pk, Rcom, p_tx=np.array([0.0, 0.0])):
    v = pk - p_tx 
    norm_pk = np.linalg.norm(v)
    if norm_pk > Rcom:
        return (Rcom / norm_pk) * v + p_tx
    else:
        return pk

def compute_relay_point(pj, ps_k, pk, pr, j_idx, Rcom):
    """
    Compute the relay point for agent j_idx:
    - Project pj onto the line spanned by ps_k and pr to get bar_pj.
    - The relay point q_j lies on the segment from pj to bar_pj.
    - If the agent can reach bar_pj before the message arrives, set q_j = bar_pj.
    - Otherwise, set q_j somewhere along [pj, bar_pj] such that
      norm(q_j - pj) = D + norm(q_j - ps_k) - j_idx*Rcom,
      where D = norm(ps_k - pk).
    """
    # Unit direction from ps_k to pr
    u = pr - ps_k
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-12:
        u = np.zeros_like(u)
    else:
        u = u / norm_u

    # Project pj onto the line spanned by ps_k and pr
    alpha = np.dot(pj - ps_k, u)
    bar_pj = ps_k + alpha * u

    # Vector from pj to bar_pj
    v = bar_pj - pj
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        direction = u  # fallback: use line direction
    else:
        direction = v / v_norm

    D = np.linalg.norm(ps_k - pk)  # distance retrieving agent to retrieval point

    # The required distance along [pj, bar_pj]
    # norm(q_j - pj) = D + norm(q_j - ps_k) - j_idx*Rcom
    # Let q_j = pj + t * direction, t in [0, v_norm]
    # norm(q_j - ps_k) = norm((pj - ps_k) + t*direction)
    # Let a = norm(pj - ps_k)
    # Let b = dot(pj - ps_k, direction)
    # norm(q_j - ps_k) = sqrt(a^2 + 2*b*t + t^2)
    # So: t = D + sqrt(a^2 + 2*b*t + t^2) - j_idx*Rcom
    # Rearranged: t + j_idx*Rcom - D = sqrt(a^2 + 2*b*t + t^2)
    # Square both sides:
    # (t + j_idx*Rcom - D)^2 = a^2 + 2*b*t + t^2
    # t^2 + 2t(j_idx*Rcom-D) + (j_idx*Rcom-D)^2 = a^2 + 2*b*t + t^2
    # 2t(j_idx*Rcom-D) + (j_idx*Rcom-D)^2 = a^2 + 2*b*t
    # 2t(j_idx*Rcom-D-b) = a^2 - (j_idx*Rcom-D)^2
    # t = [a^2 - (j_idx*Rcom-D)^2] / [2*(j_idx*Rcom-D-b)]   (if denominator != 0)

    a = np.linalg.norm(pj - ps_k)
    b = np.dot(pj - ps_k, direction)
    offset = j_idx * Rcom - D
    denom = 2 * (offset - b)
    numer = a**2 - offset**2

    if np.abs(denom) < 1e-12:
        t = 0.0
    else:
        t = numer / denom

    # Clamp t to [0, v_norm]
    t = np.clip(t, 0, v_norm)

    # Compute relay point
    q_j = pj + t * direction

    return q_j

def dijkstra_baseline(p_agents, p_tx, p_recv, Rcom=1.0, sigma=0.1, beta=0.99, c_pos=0.1):
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
        ps_k = compute_retrieval_point(pk, Rcom, p_tx)
        pr = p_recv

        # Build graph G_k
        Gk = nx.DiGraph()

        # Node set
        node_map = {}  # maps agent idx to relay point
        Gk.add_node(k, pos=pk)  # retrieving agent
        Gk.add_node('s_k', pos=ps_k)
        Gk.add_node('r', pos=pr)

        # Direction vector from retrieval point to receiver
        direction = pr - ps_k
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-12:
            direction = np.zeros_like(direction)
        else:
            direction = direction / norm_dir

        # Sort other agents by their projection onto the line from ps_k to pr
        agent_alphas = []
        for j in I:
            if j == k:
                continue
            pj = p_agents[j]
            # Project agent onto the line from ps_k to pr
            alpha = np.dot(pj - ps_k, direction)
            if alpha >= 0:
                agent_alphas.append((j, alpha))
        # Sort by increasing alpha (distance along the line from ps_k)
        sorted_agents = [j for j, alpha in sorted(agent_alphas, key=lambda x: x[1])]

        # Add relay agents
        for j in sorted_agents:

            # No need to add retrieving agent again
            if j == k:  
                continue

            # Compute relay point for agent j
            pj = p_agents[j]
            relay_j = compute_relay_point(pj, ps_k, pk, pr, j_idx=(j+1), Rcom=Rcom)  # \hat{p}_j in paper  OBS! Bug: j_idx assumes agents are ordered..
            node_map[j] = relay_j
            Gk.add_node(j, pos=relay_j)

        # Edge set
        # From retrieving agent k to its retrieval point s_k
        Gk.add_edge(k, 's_k', weight=np.linalg.norm(ps_k - pk))

        # From retrieval point s_k to receiving base r
        Gk.add_edge('s_k', 'r', weight=max(0, np.linalg.norm(pr - ps_k) - Rcom))

        # From retrieval point s_k to relay agents j
        for j, relay_j in node_map.items():
            if j != k:
                w = max(0, np.linalg.norm(relay_j - ps_k) - Rcom)
                Gk.add_edge('s_k', j, weight=w)

                # From relay agents j to receiving base r
                Gk.add_edge(j, 'r', weight=max(0, np.linalg.norm(pr - relay_j) - Rcom))
        
        # From relay agents j to other relay agents l
        for j, relay_j in node_map.items():
            if j != k:
                for l, relay_l in node_map.items():
                    if j != l and l != k:
                        Gk.add_edge(j, l, weight=max(0, np.linalg.norm(relay_l - relay_j) - Rcom))

        # Run Dijkstra from k to r
        try:
            path = nx.dijkstra_path(Gk, source=k, target='r', weight='weight')

            # Compute total distance of path
            total_dist = 0
            for i in range(len(path)-1):
                total_dist += Gk.edges[path[i], path[i+1]]['weight']


            candidate_paths.append((path, total_dist))
            candidate_relay_points.append(node_map)
            candidate_graphs.append(Gk)
            candidate_k.append(k)

            # Update best if needed
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
    k_D = len(relaying_agents)                 # number of relaying agents
    T_M = int(np.ceil(best_distance / sigma))  # movement time
    T_D = T_M + k_D + 2                        # total delivery time

    # Compute per-agent distances and movement times
    D_k = np.zeros(K)
    t_start_k = np.zeros(K, dtype=int)
    t_stop_k = -1*np.ones(K, dtype=int)  # -1 means agent does not move

    # Map agent index to relay point in path
    relay_points_in_path = []
    for n in best_path:
        if isinstance(n, int) and n != best_k:
            relay_points_in_path.append(best_relay_points[n])

    # Distance and movement times for retrieving agent
    if k_D > 0:  # there are relaying agents
        first_relay = relaying_agents[0]
        t_stop_k[best_k] = int(np.ceil(
            np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) / sigma
        )) + 1 + int(np.ceil(
            max(0, np.linalg.norm(best_relay_points[first_relay] - best_graph.nodes['s_k']['pos']) - Rcom
        ) / sigma))
    else:
        # No relaying agents: retrieving agent goes to retrieval point, then to receiver
        t_stop_k[best_k] = int(np.ceil(
            np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) / sigma
        )) + 1 + int(np.ceil(
            max(0, np.linalg.norm(p_recv - best_graph.nodes['s_k']['pos']) - Rcom
        ) / sigma))

    # Distances and movement times for relaying agents
    for idx, k in enumerate(relaying_agents):
        relay_point = best_relay_points[k]
        if idx < k_D - 1:  # not the last relay agent
            next_relay = relaying_agents[idx+1]
            D_k[k] = np.linalg.norm(relay_point - p_agents[k]) \
                + max(0, np.linalg.norm(best_relay_points[next_relay] - relay_point) - Rcom)
        else:  # last relay agent
            D_k[k] = np.linalg.norm(relay_point - p_agents[k]) \
                + max(0, np.linalg.norm(p_recv - relay_point) - Rcom)
            
    # Movement times
    for idx, k in enumerate(relaying_agents):
        # Start time: must arrive at relay point before the preceding agent arrives
        travel_projection_time = int(np.ceil(np.linalg.norm(best_relay_points[k] - p_agents[k]) / sigma))
        if idx == 0:
            t_start_k[k] = t_stop_k[best_k] - travel_projection_time
        else:
            prev_relay = relaying_agents[idx-1]
            prev_arrival = t_stop_k[prev_relay]  # previous relay agent's arrival at this relay point
            t_start_k[k] = prev_arrival - travel_projection_time

        # Stop time: when agent reaches the point where it relays the message to the next agent
        next_relay_time = int(np.ceil(max(0, np.linalg.norm(best_relay_points[relaying_agents[idx+1]] - best_relay_points[k]) - Rcom) / sigma)) \
            if idx < k_D - 1 else int(np.ceil(max(0, np.linalg.norm(p_recv - best_relay_points[k]) - Rcom) / sigma))
        t_stop_k[k] = t_start_k[k] + travel_projection_time + 1 + next_relay_time  

    # Value computation (see reasoning behind cost parameters, especially state s^#)
    T = int(np.floor((1.1 * R + 2 * Rcom) / sigma)) + K
    Rmax = (K+4)*Rcom
    D_tot = (1 + 0.1 * K) * Rmax + (2 + 0.5 * K * (K - 1)) * Rcom
    r_del = ((1-beta**T)/(1-beta))*(D_tot/T)
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
    agent_radiuses = Ra * np.sqrt(rng.uniform(0, 1, size=(K, 1)))    
    agent_angles = rng.uniform(0, 2*np.pi, size=(K, 1))
    p_agents = p0 + agent_radiuses*np.hstack([np.cos(agent_angles), np.sin(agent_angles)])

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
    ax.scatter(*p_tx, c='green', marker='s', label='Transmitting base')
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
    ax.legend(loc="upper left")

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

    # Colors and line widths
    agent_color = "orange"
    init_agent_color = "grey"
    receive_color = "blue"
    deliver_color = "green"
    passive_color = "red"
    alpha_circle = 0.6
    alpha_init_agent = 0.6
    line_width = 3
    marker_size = 100

    fig, ax = plt.subplots(figsize=(6, 6))

    # Transmitting base
    ax.scatter(*p_tx, c=receive_color, marker='s', s=marker_size, label='TX base')
    ax.add_artist(plt.Circle(p_tx, Rcom, color=receive_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Midpoint base p0 radius (where agents are initialized)
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Receiver base
    ax.scatter(*p_recv, c=deliver_color, marker='s', s=marker_size, label='RX base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Node positions
    node_positions = {'s': p_tx, 'r': p_recv}
    for i, pos in enumerate(p_agents):
        node_positions[i] = pos

    # Add retrieval point for the retrieving agent
    km = dijkstra_result['retrieving_agent']
    ps_k = compute_retrieval_point(p_agents[km], sample['Rcom'])
    node_positions['s_k'] = ps_k

    # Plot dashed line from retrieval point to the Rcom circle of the receiving base
    direction = p_recv - ps_k
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
        end_point = p_recv - direction * Rcom
    else:
        end_point = p_recv
    ax.plot([ps_k[0], end_point[0]], [ps_k[1], end_point[1]],
            c=init_agent_color, linestyle='--', linewidth=line_width, alpha=0.6, zorder=1)

    # Instead of plotting the retrieving point, plot the point where the retrieving agent relays to the next agent:
    if dijkstra_result['relaying_agents']:
        first_relay = dijkstra_result['relaying_agents'][0]
        start = ps_k
        end = dijkstra_result['relay_points'][first_relay]
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
            relay_contact = end - direction * sample['Rcom']
            ax.scatter(relay_contact[0], relay_contact[1], c=agent_color, marker='o', s=marker_size)

    # Plot the retrieval points
    #ax.scatter(ps_k[0], ps_k[1], c=agent_color, marker='o')


    path = dijkstra_result['path']
    relay_points = dijkstra_result['relay_points']

    # Track if relay points and passive agents are plotted for legend
    relay_point_plotted = False
    passive_agent_plotted = False

    # Plot relay points (destinations) with full opacity, only for relaying agents
    for idx in dijkstra_result['relaying_agents']:
        relay = relay_points[idx]
        ax.scatter(relay[0], relay[1], c=agent_color, alpha=1.0, marker='o', s=marker_size, zorder=20)
        # Add Rcom circle around relay point
        circle = plt.Circle(relay, Rcom, color=agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
        ax.add_artist(circle)
        relay_point_plotted = True

    # Plot initial agent positions
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size, zorder=21)
    
    # Plot passive agents in red at their initial positions
    for idx in range(len(p_agents)):
        if idx not in dijkstra_result['relaying_agents'] and idx != dijkstra_result['retrieving_agent']:
            ax.scatter(p_agents[idx, 0], p_agents[idx, 1], c=passive_color, alpha=1.0, marker='o', s=marker_size, zorder=22)

    # Connect only relaying agents' initial positions to their relay points with dashed lines
    for idx, relay in relay_points.items():
        if idx in dijkstra_result['relaying_agents']:
            agent_pos = p_agents[idx]
            ax.annotate(
                "",  # No text
                xy=relay,
                xytext=agent_pos,
                arrowprops=dict(
                    arrowstyle="->",
                    color=init_agent_color,
                    lw=line_width,
                    alpha=alpha_init_agent
                )
            )

    # Message retrieval arrow (agent -> transmitter)
    #ax.annotate("", xy=node_positions['s'], xytext=node_positions[km],
    #            arrowprops=dict(arrowstyle="->", color=receive_color, lw=2))

    # Message delivery arrows along Dijkstra path (km -> ... -> r)
    last_meet_point = None  # Will store the final relay point on Rcom of receiver
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
                end = relay_points[node_to] - direction * sample['Rcom']
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))
        elif node_to == 'r':
            # Last arrow: from last relay point to edge of receiver base Rcom circle
            direction = p_recv - start
            norm = np.linalg.norm(direction)
            if norm > Rcom + 1e-8:
                direction = direction / norm
                end = p_recv - direction * sample['Rcom']
                last_meet_point = end  # Save the final relay point
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))
        else:
            # For other cases, plot arrow if distance > Rcom
            norm = np.linalg.norm(end - start)
            if norm > Rcom + 1e-8:
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))

    # Plot the relaying point where the last agent meets Rcom of the receiving base
    if last_meet_point is not None:
        ax.scatter(last_meet_point[0], last_meet_point[1], c=agent_color, marker='o', s=marker_size)# label='Final relay point')
        
    # Plot initial agent positions
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size)#, label='Init pos')

    # Adjust plot limits
    ax.set_xlim(-0.2*Ra, R+0.2*Ra)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)

    # Annotations
    #ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(len(p_agents)+1)*Rcom:.3f}",
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Custom legend for arrows and relay points
    legend_elements = [
        Line2D([0], [0], color=deliver_color, lw=2, label='Relay path'),
        Line2D([0], [0], marker='o', color='w', label='Init pos', markerfacecolor=init_agent_color, markersize=10, alpha=alpha_init_agent),
        Line2D([0], [0], marker='o', color='w', label='Relay point', markerfacecolor=agent_color, markersize=10),
    ]
    # Only add passive agent if it was plotted
    if passive_agent_plotted:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Passive agent', markerfacecolor=passive_color, markersize=10))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + [le.get_label() for le in legend_elements], loc="upper left")

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.title(
        "Dijkstra Baseline\n"
        f"($K$: {K}; Time steps: {dijkstra_result['delivery_time']}; Value: {dijkstra_result['value']:.2f})"
    )


    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


def animate_scenario_with_path_colored(sample, dijkstra_result, savepath=None, interval=100):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    p_agents = sample['p_agents']
    K = p_agents.shape[0]
    t_start_k = dijkstra_result['agent_start_times']
    t_stop_k = dijkstra_result['agent_stop_times']
    relay_points = dijkstra_result['relay_points']
    relaying_agents = dijkstra_result['relaying_agents']
    retrieving_agent = dijkstra_result['retrieving_agent']

    # Colors and line widths
    agent_color = "orange"
    init_agent_color = "grey"
    send_color = "blue"
    deliver_color = "green"
    passive_color = "red"
    alpha_circle = 0.6
    alpha_init_agent = 0.6
    line_width = 3
    marker_size = 100

    R = sample['R']
    Rcom = sample['Rcom']
    Ra = sample['Ra']
    p_tx = sample['p_tx']
    p_recv = sample['p_recv']

    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot initial agent positions in grey (static, underneath everything)
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size, zorder=1, label='Init pos')

    # --- Static elements ---
    ax.scatter(*p_tx, c=send_color, marker='s', s=marker_size, label='TX base')
    ax.add_artist(plt.Circle(p_tx, Rcom, color=send_color, fill=False, linestyle='--', lw=line_width, alpha=alpha_circle))
    ax.scatter(*p_recv, c=deliver_color, marker='s', s=marker_size, label='RX base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--', lw=line_width, alpha=alpha_circle))
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Add retrieval point for the retrieving agent
    km = retrieving_agent
    ps_k = compute_retrieval_point(p_agents[km], sample['Rcom'])

    # Plot dashed line from retrieval point to the Rcom circle of the receiving base
    direction = p_recv - ps_k
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
        end_point = p_recv - direction * Rcom
    else:
        end_point = p_recv
    ax.plot([ps_k[0], end_point[0]], [ps_k[1], end_point[1]],
            c=init_agent_color, linestyle='--', linewidth=line_width, alpha=0.6, zorder=1)

    # --- 1) Build waypoints for each agent ---
    waypoints = [None] * K
    for k in range(K):
        if k == retrieving_agent:
            # Retrieving agent: initial -> retrieval point -> first relay point or receiver Rcom
            w = [p_agents[k], ps_k]
            if relaying_agents:
                first_relay = relaying_agents[0]
                relay_dest = relay_points[first_relay]
                direction = relay_dest - ps_k
                dist = np.linalg.norm(direction)
                if dist > Rcom:
                    direction = direction / dist
                    relay_dest = relay_dest - direction * Rcom
                w.append(relay_dest)
            else:
                direction = p_recv - ps_k
                dist = np.linalg.norm(direction)
                if dist > 1e-8:
                    direction = direction / dist
                    relay_dest = p_recv - direction * Rcom
                else:
                    relay_dest = p_recv
                w.append(relay_dest)
            waypoints[k] = w
        elif k in relay_points:
            if k in relaying_agents:
                idx_in_chain = relaying_agents.index(k)
                w = [p_agents[k]]
                relay_with_prev = relay_points[k]
                w.append(relay_with_prev)
                if idx_in_chain < len(relaying_agents) - 1:
                    next_agent = relaying_agents[idx_in_chain + 1]
                    relay_with_next = relay_points[next_agent]
                    direction = relay_with_next - relay_with_prev
                    dist = np.linalg.norm(direction)
                    if dist > Rcom:
                        direction = direction / dist
                        relay_with_next = relay_with_next - direction * Rcom
                    w.append(relay_with_next)
                else:  # last relay agent: relay to receiver Rcom
                    direction = p_recv - relay_with_prev
                    dist = np.linalg.norm(direction)
                    if dist > Rcom:
                        direction = direction / dist
                        relay_with_next = p_recv - direction * Rcom
                        w.append(relay_with_next)
                waypoints[k] = w
            else:
                waypoints[k] = [p_agents[k]]
        else:
            waypoints[k] = [p_agents[k]]

    # --- 2) Agent position interpolation ---
    def get_agent_pos(k, t):
        t_start = t_start_k[k]
        t_stop = t_stop_k[k]
        wp = waypoints[k]
        n_wp = len(wp)
        if n_wp == 1:
            return wp[0]
        elif n_wp == 2:
            seg1 = np.linalg.norm(wp[1] - wp[0])
            if seg1 == 0:
                return wp[0]
            if t < t_start:
                return wp[0]
            elif t >= t_stop:
                return wp[1]
            else:
                frac = (t - t_start) / max(1, t_stop - t_start)
                dist = frac * seg1
                return wp[0] + (wp[1] - wp[0]) * (dist / seg1)
        elif n_wp == 3:
            seg1 = np.linalg.norm(wp[1] - wp[0])
            seg2 = np.linalg.norm(wp[2] - wp[1])
            total = seg1 + seg2
            total_time = t_stop - t_start
            if t < t_start:
                return wp[0]
            elif t >= t_stop:
                return wp[2]
            else:
                frac = (t - t_start) / max(1, total_time)
                dist = frac * total
                if dist <= seg1:
                    if seg1 == 0:
                        return wp[1]
                    return wp[0] + (wp[1] - wp[0]) * (dist / seg1)
                else:
                    if seg2 == 0:
                        return wp[2]
                    return wp[1] + (wp[2] - wp[1]) * ((dist - seg1) / seg2)
        else:
            return wp[-1]

    # --- 3) Animation objects ---
    agent_dots = ax.scatter(p_agents[:, 0], p_agents[:, 1], c=[agent_color]*K, s=marker_size, zorder=30)
    relay_arrows1 = []
    relay_arrows2 = []
    for _ in relaying_agents:
        arrow1 = ax.annotate("", xy=(0,0), xytext=(0,0),
                             arrowprops=dict(arrowstyle="->", color=init_agent_color, lw=line_width, alpha=alpha_init_agent))
        relay_arrows1.append(arrow1)
        arrow2 = ax.annotate("", xy=(0,0), xytext=(0,0),
                             arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width, alpha=1.0))
        relay_arrows2.append(arrow2)
    retrieving_arrow1 = ax.annotate("", xy=(0,0), xytext=(0,0),
                                    arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width, alpha=1.0))
    retrieving_arrow2 = ax.annotate("", xy=(0,0), xytext=(0,0),
                                    arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width, alpha=1.0))

    # --- 4) Animation update function ---
    def animate(t):
        # Remove previous Rcom circles
        for c in getattr(ax, "_agent_circles", []):
            try:
                c.remove()
            except Exception:
                pass
        ax._agent_circles = []

        current_pos = np.zeros_like(p_agents)
        colors = []
        for k in range(K):
            pos = get_agent_pos(k, t)
            current_pos[k] = pos
            if k == retrieving_agent or k in relaying_agents:
                if t < t_stop_k[k]:
                    colors.append(agent_color)
                else:
                    colors.append(deliver_color)
                # Draw Rcom circle only while moving
                if t < t_stop_k[k]:
                    circ = plt.Circle(pos, Rcom, color=agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
                    ax.add_artist(circ)
                    ax._agent_circles.append(circ)
            else:
                colors.append(passive_color)
                # No Rcom circle for passive agents

        agent_dots.set_offsets(current_pos)
        agent_dots.set_facecolors(colors)

        # --- Relaying agent arrows (two segments) ---
        for idx, k in enumerate(relaying_agents):
            wp = waypoints[k]
            pos = get_agent_pos(k, t)
            if len(wp) == 3:
                seg1 = np.linalg.norm(wp[1] - wp[0])
                seg2 = np.linalg.norm(wp[2] - wp[1])
                total = seg1 + seg2
                t_start = t_start_k[k]
                t_stop = t_stop_k[k]
                total_time = t_stop - t_start
                if t < t_start:
                    frac = 0
                elif t >= t_stop:
                    frac = 1
                else:
                    frac = (t - t_start) / max(1, total_time)
                dist = frac * total
                if dist <= seg1:
                    # On first segment: initial -> relay_with_prev
                    relay_arrows1[idx].set_position(wp[0])
                    relay_arrows1[idx].xy = pos
                    relay_arrows1[idx].arrow_patch.set_color(init_agent_color)
                    relay_arrows1[idx].arrow_patch.set_alpha(alpha_init_agent)
                    relay_arrows2[idx].set_position((0,0))
                    relay_arrows2[idx].xy = (0,0)
                    relay_arrows2[idx].arrow_patch.set_alpha(0)
                else:
                    # On second segment: show both arrows
                    relay_arrows1[idx].set_position(wp[0])
                    relay_arrows1[idx].xy = wp[1]
                    relay_arrows1[idx].arrow_patch.set_color(init_agent_color)
                    relay_arrows1[idx].arrow_patch.set_alpha(alpha_init_agent)
                    relay_arrows2[idx].set_position(wp[1])
                    relay_arrows2[idx].xy = pos
                    relay_arrows2[idx].arrow_patch.set_color(deliver_color)
                    relay_arrows2[idx].arrow_patch.set_alpha(1)
            elif len(wp) == 2:
                relay_arrows1[idx].set_position(wp[0])
                relay_arrows1[idx].xy = pos
                relay_arrows1[idx].arrow_patch.set_color(init_agent_color)
                relay_arrows1[idx].arrow_patch.set_alpha(alpha_init_agent)
                relay_arrows2[idx].set_position((0,0))
                relay_arrows2[idx].xy = (0,0)
                relay_arrows2[idx].arrow_patch.set_alpha(0)

        # --- Retrieving agent arrows (two segments) ---
        wp = waypoints[retrieving_agent]
        pos = get_agent_pos(retrieving_agent, t)
        if len(wp) == 3:
            seg1 = np.linalg.norm(wp[1] - wp[0])
            seg2 = np.linalg.norm(wp[2] - wp[1])
            total = seg1 + seg2
            t_start = t_start_k[retrieving_agent]
            t_stop = t_stop_k[retrieving_agent]
            total_time = t_stop - t_start
            if t < t_start:
                frac = 0
            elif t >= t_stop:
                frac = 1
            else:
                frac = (t - t_start) / max(1, total_time)
            dist = frac * total
            if dist <= seg1:
                retrieving_arrow1.set_position(wp[0])
                retrieving_arrow1.xy = pos
                retrieving_arrow1.arrow_patch.set_color(deliver_color)
                retrieving_arrow1.arrow_patch.set_alpha(1)
                retrieving_arrow2.set_position((0,0))
                retrieving_arrow2.xy = (0,0)
                retrieving_arrow2.arrow_patch.set_alpha(0)
            else:
                retrieving_arrow1.set_position(wp[0])
                retrieving_arrow1.xy = wp[1]
                retrieving_arrow1.arrow_patch.set_color(deliver_color)
                retrieving_arrow1.arrow_patch.set_alpha(1)
                retrieving_arrow2.set_position(wp[1])
                retrieving_arrow2.xy = pos
                retrieving_arrow2.arrow_patch.set_color(deliver_color)
                retrieving_arrow2.arrow_patch.set_alpha(1)
        elif len(wp) == 2:
            retrieving_arrow1.set_position(wp[0])
            retrieving_arrow1.xy = pos
            retrieving_arrow1.arrow_patch.set_color(deliver_color)
            retrieving_arrow1.arrow_patch.set_alpha(1)
            retrieving_arrow2.set_position((0,0))
            retrieving_arrow2.xy = (0,0)
            retrieving_arrow2.arrow_patch.set_alpha(0)

        ax.set_title(f"Time: {t} / {int(np.max(t_stop_k)) + 1}")

        return [agent_dots] + relay_arrows1 + relay_arrows2 + [retrieving_arrow1, retrieving_arrow2]

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='TX base', markerfacecolor=send_color, markersize=10),
        Line2D([0], [0], marker='s', color='w', label='RX base', markerfacecolor=deliver_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Active agent', markerfacecolor=agent_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Passive agent', markerfacecolor=passive_color, markersize=10),
        Line2D([0], [0], color=deliver_color, lw=2, label='Relay path'),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_aspect('equal')
    ax.set_xlim(-0.4 * Ra, R + 0.4 * Ra)
    ax.set_ylim(-1.3 * Ra, 1.3 * Ra)
    

    ani = animation.FuncAnimation(
        fig, animate, frames=int(np.max(t_stop_k)) + 2, interval=interval, blit=False, repeat=False
    )

    if savepath:
        ani.save(savepath, writer='pillow', fps=1000//interval)
    else:
        plt.show()


if __name__ == '__main__':
    # Scenario parameters
    K = 1  # something weird with K=3 seed=1  (weird meetup point for retrieving agent and first relay agent)
    Rcom = 1.0
    R = (K+4)*Rcom
    Ra = 0.6*R

    sigma = 0.1
    beta = 0.99

    seed=1

    # Cost parameters
    cost_motion = 5
    cost_antenna = 1

    # Saving stuff
    save_path = fr'Plots/dijkstra_baseline_K{K}_seed{seed}.png'
    animation_save_path = fr'Animations/dijkstra_baseline_animation_K{K}_seed{seed}.gif'

    sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, seed=seed)
    #plot_scenario(sample, savepath=save_path)
    result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom, sigma=sigma, beta=beta, c_pos=cost_motion)    
    #plot_scenario_with_path_colored(sample, result, savepath=save_path)
    animate_scenario_with_path_colored(sample, result, savepath=animation_save_path)