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
    # Compute unit direction u from retrieving point to receiving base
    u = pr - ps_k
    norm_u = np.linalg.norm(u)
    if norm_u == 0:  # will never happen in our scenarios
        u = np.zeros_like(u)
    else:
        u = u / norm_u

    # Orthogonal projection of pj (relay agent j's initial position) onto line 
    alpha_j = np.dot(pj - ps_k, u) 
    pj_bar = ps_k + alpha_j * u  # projected point on line

    # Help parameters 
    b = np.linalg.norm(pj - pj_bar)  # distance from pj to line
    D = np.linalg.norm(ps_k - pk)    # distance between retrieving agent and its retrieval point
    a = np.linalg.norm(pj_bar - ps_k) - (j_idx-1)*Rcom  # distance from pj_bar to ps_k minus offset

    # Condition for direct projection 
    cond = b <= D - a  # (original equation)
    if cond:  # relay agent can reach line in time 
        return pj_bar
    else:
        # Fallback: solve for q_j 
        n = (pj - pj_bar) / (b + 1e-12)  # unit normal from line to pj

        # Quadratic coefficients
        A = 4*(b**2 - D**2)
        B = 4*b*(D**2 + a**2 - b**2)
        C = (D**2 + a**2 - b**2)**2 - 4*D**2*a**2

        discriminant = B**2 - 4*A*C
        if discriminant < 0 or A == 0:
            return pj_bar  # fallback to projection

        t1 = (-B + np.sqrt(discriminant)) / (2*A)
        t2 = (-B - np.sqrt(discriminant)) / (2*A)

        # Candidates
        q1 = pj_bar + t1 * n
        q2 = pj_bar + t2 * n

        # Admissible root: same side as pj and original equation holds
        candidates = []
        for t, q in zip([t1, t2], [q1, q2]):
            if np.dot(q - pj_bar, pj - pj_bar) >= 0:
                # Check original equation 
                lhs = np.abs(b - t)             # distance traveled by relay agent
                rhs = D + np.sqrt(a**2 + t**2)  # distance traveled by retrieving agent
                if np.isclose(lhs, rhs, atol=1e-6):
                    candidates.append(q)
        if candidates:
            return candidates[0]
        else:
            return pj_bar

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

        # Add relay agents
        for j in I:

            # No need to add retrieving agent again
            if j == k:  
                continue

            # Compute relay point for agent j
            pj = p_agents[j]
            relay_j = compute_relay_point(pj, ps_k, pk, pr, j_idx=(j+1), Rcom=Rcom)  # \hat{p}_j in paper
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

        # Use retrieving agent's position and retrieval point
        D_k[best_k] = np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) \
            + max(0, np.linalg.norm(best_relay_points[first_relay] - best_graph.nodes['s_k']['pos']) - Rcom)
    else:  # no relaying agents
        D_k[best_k] = max(0, np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) - Rcom)
    
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
    t_stop_k[best_k] = int(np.ceil(
                np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) / sigma
            )) + 1 + int(np.ceil(  # +1 for relay time
                max(0, np.linalg.norm(best_relay_points[0] - best_graph.nodes['s_k']['pos']) - Rcom
            ) / sigma))  
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

    # Colors and line widths
    agent_color = "blue"
    init_agent_color = "grey"
    receive_color = "red"
    deliver_color = "green"
    alpha_circle = 0.6
    alpha_init_agent = 0.6
    line_width = 2

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot initial agent positions
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent)

    # Node positions
    node_positions = {'s': p_tx, 'r': p_recv}
    for i, pos in enumerate(p_agents):
        node_positions[i] = pos

    # Add retrieval point for the retrieving agent
    km = dijkstra_result['retrieving_agent']
    ps_k = compute_retrieval_point(p_agents[km], sample['Rcom'])
    node_positions['s_k'] = ps_k

    # Plot the retrieval points
    ax.scatter(ps_k[0], ps_k[1], c=agent_color, marker='o')

    # Plot dashed line between retrieval point and receiving base
    ax.plot([ps_k[0], p_recv[0]], [ps_k[1], p_recv[1]], c=init_agent_color, linestyle='--', linewidth=line_width, alpha=0.6)#, label='Relay projection line')

    path = dijkstra_result['path']
    relay_points = dijkstra_result['relay_points']

    # Plot relay points (destinations) with full opacity
    for idx, relay in relay_points.items():
        ax.scatter(relay[0], relay[1], c=agent_color, alpha=1.0, marker='o', label='Relay agent' if idx == list(relay_points.keys())[0] else "")
        # Add blue circle of radius Rcom around each relay point
        circle = plt.Circle(relay, Rcom, color=agent_color, fill=False, linestyle='--', linewidth=line_width , alpha=alpha_circle)
        ax.add_artist(circle)

    # Connect initial agent positions to their relay points with dashed lines
    for idx, relay in relay_points.items():
        agent_pos = p_agents[idx]
        ax.annotate(
            "",  # No text
            xy=relay,
            xytext=agent_pos,
            arrowprops=dict(
                arrowstyle="->",
                color="grey",
                lw=line_width,
                alpha=alpha_init_agent
            )
)

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
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))
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

    # Transmitting base
    ax.scatter(*p_tx, c=receive_color, marker='s', label='Transmitting base')
    ax.add_artist(plt.Circle(p_tx, Rcom, color=receive_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Midpoint base p0
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Receiver base
    ax.scatter(*p_recv, c=deliver_color, marker='s', label='Receiver base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Adjust plot limits
    ax.set_xlim(-1.3*Rcom, R+1.3*Rcom)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)

    # Annotations
    #ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(len(p_agents)+1)*Rcom:.3f}",
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Custom legend for arrows and relay points
    legend_elements = [
        Line2D([0], [0], color=deliver_color, lw=2, label='Relaying path'),
        Line2D([0], [0], marker='o', color='w', label='Initial agent positions', markerfacecolor=init_agent_color, alpha=alpha_init_agent)
        #Line2D([0], [0], marker='o', color='w', label='Relay point', markerfacecolor='orange', markersize=8),
        #Line2D([0], [0], linestyle='--', color='k', label='Agent to relay', alpha=0.7)
    ]
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
        ax.scatter(*p_tx, c=receive_color, marker='s', label='Transmitting base')
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
    

def animate_dijkstra_minimal(sample, dijkstra_result, animation_save_path):
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
    ani.save(animation_save_path, writer='pillow', dpi=120)
    plt.close(fig)
    


if __name__ == '__main__':
    # Scenario parameters
    K = 20
    Rcom = 1.0
    R = (K+4)*Rcom
    Ra = 0.6*R

    sigma = 0.1
    beta = 0.99

    seed=3

    # Cost parameters
    cost_motion = 5
    cost_antenna = 1

    # Saving stuff
    save_path = fr'Plots/dijkstra_baseline_K{2}_seed{5}.png'
    animation_save_path = fr'Animations/dijkstra_baseline_animation_K{K}_seed{seed}.gif'

    sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, seed=seed)
    #plot_scenario(sample, savepath=save_path)
    result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom, sigma=sigma, beta=beta, c_pos=cost_motion)    
    plot_scenario_with_path_colored(sample, result, savepath=save_path)

    # Uncomment to generate animation
    #animate_dijkstra(sample, result, savepath='dijkstra_animation.mp4')
    #animate_dijkstra_minimal(sample, result, animation_save_path)