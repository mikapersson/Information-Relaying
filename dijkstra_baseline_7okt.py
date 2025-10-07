import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os

"""
More complex:
Assumes the edge weight for ps_k -> q_j takes all possible paths to q_j into consideration.
"""

def compute_retrieval_point(pk, Rcom, p_tx=np.array([0.0, 0.0])):
    v = pk - p_tx 
    norm_pk = np.linalg.norm(v)
    if norm_pk > Rcom:
        return (Rcom / norm_pk) * v + p_tx
    else:
        return pk

def compute_relay_point(pj, ps_k, pk, pr, j_k, Rcom):
    """
    Compute the relay point for agent j_k (index in sorted projection order, starting from 1):
    - Project pj onto the line from ps_k to pr to get bar_pj (pmeet_j).
    - If agent can reach bar_pj in time, return bar_pj.
    - Otherwise, return q_j as described in the prompt.
    """
    # Direction from ps_k to pr
    u = pr - ps_k
    norm_u = np.linalg.norm(u)
    u = u / norm_u

    # Projection of pj onto the line (pmeet_j)
    bar_pj = ps_k + np.dot(pj - ps_k, u) * u

    a = np.linalg.norm(bar_pj - pj)
    c = np.linalg.norm(ps_k - pk)
    d = np.linalg.norm(bar_pj - ps_k)
    #a = np.linalg.norm(bar_pj - ps_k) - j_k*Rcom

    # Condition for using the projection point (agent makes it to its projection before the message does)
    if a <= c + max(0, d - j_k*Rcom):  # lambda <= 0
        return bar_pj

    # Agent doesn't make it to its projection before the message does
    # v = unit normal from line to pj
    diff = pj - bar_pj
    norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-12:
        # pj is on the line, but can't reach bar_pj in time, so just return bar_pj
        return bar_pj
    v = diff / norm_diff

    # Test lambda = a - c
    lam = a - c  # if lambda <= 0, relaying agent reaches bar_pj before the message does
    if lam >= 0 and np.sqrt(d**2 + lam**2) <= j_k*Rcom:  
        qj = bar_pj + lam * v
        return qj
    else:  
        A = a - c + j_k*Rcom 
        if np.isclose(A, 0):
            return bar_pj
            #raise ValueError("Relay point computation error: A is zero")
        else:
            lam = (A**2 - d**2) / (2*A)
            if lam >= 0 and np.sqrt(d**2 + lam**2) > j_k*Rcom:
                qj = bar_pj + lam * v
                return qj
    

def sample_capsule_point(R, Rcom, rng):
    """
    Sample a point uniformly from a capsule (rectangle + two half circles) along the x-axis,
    with rectangle width R, height 3*Rcom, and half circle radii 1.5*Rcom.
    The left circle is centered at (0, 0), the right at (R, 0).
    The rectangle's left side intersects the origin and has width R.
    """
    height = 3 * Rcom
    radius = 1.5 * Rcom
    area_rect = R * height
    area_circ = np.pi * radius**2
    area_total = area_rect + area_circ

    u = rng.uniform(0, area_total)
    if u < area_rect:
        # Rectangle part: x in [0, R]
        x = rng.uniform(0, R)
        y = rng.uniform(-height/2, height/2)
    else:
        # One of the half circles
        u_circ = rng.uniform(0, 1)
        r = radius * np.sqrt(u_circ)
        if rng.uniform() < 0.5:
            # Left half circle (centered at (0,0)), theta in [-pi/2, pi/2]
            theta = rng.uniform(np.pi/2, 3*np.pi/2)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        else:
            # Right half circle (centered at (R,0)), theta in [pi/2, 3pi/2]
            theta = rng.uniform(-np.pi/2, np.pi/2)
            x = R + r * np.cos(theta)
            y = r * np.sin(theta)
    return np.array([x, y])


def dijkstra_baseline(p_agents, p_tx, p_recv, Rcom=1.0, sigma=0.1, beta=0.99, c_pos=0.1, jammer=False):
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
        direction = direction / norm_dir

        # Project all other agents onto the line and sort by increasing distance from ps_k to projection point
        agent_projections = []
        for j in I:
            if j == k:
                continue
            pj = p_agents[j]
            # Compute projection point of pj onto the line through ps_k and pr
            u = direction
            proj_pj = ps_k + np.dot(pj - ps_k, u) * u
            dist = np.linalg.norm(proj_pj - ps_k)
            # Only keep agents whose projection is to the right of ps_k
            if 0 <= np.dot(proj_pj - ps_k, pr - ps_k) :
                agent_projections.append((j, proj_pj, dist))

        # Sort by increasing distance from ps_k to projection point
        sorted_agents = [j for j, proj, dist in sorted(agent_projections, key=lambda x: x[2])]

        # Add relay agents and compute relay points
        for idx, j in enumerate(sorted_agents):  
            pj = p_agents[j]
            relay_j = compute_relay_point(pj, ps_k, pk, pr, j_k=idx+1, Rcom=Rcom)
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

    # TODO: Adjust for jammer 
    if jammer:
        print("Not implemented: jammer adjustment")

    # Find relaying agents in optimal path (excluding retrieving agent and bases)
    relaying_agents = [n for n in best_path if isinstance(n, int) and n != best_k]
    k_D = len(relaying_agents)                 # number of relaying agents 
    T_M = int(np.ceil(best_distance / sigma))  # movement time
    T_D = T_M + k_D + 2                        # total delivery time

    # Compute per-agent distances and movement times (used for the value computation AND animation)
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
            
    # Sanity check, no active agent has reached its relay point before the message could have (excluding air distance for the message)
    if debug:
        for idx, k in enumerate(relaying_agents):
            message_distance = np.linalg.norm(best_graph.nodes['s_k']['pos'] - p_agents[best_k]) + \
            max(0, np.linalg.norm(best_relay_points[relaying_agents[0]] - best_graph.nodes['s_k']['pos']) - Rcom)
            if idx > 0:
                message_distance += sum(
                    max(0, np.linalg.norm(best_relay_points[relaying_agents[i]] - best_relay_points[relaying_agents[i-1]]) - Rcom) for i in range(idx)  # distance to next relay agent
                )

            if np.linalg.norm(best_relay_points[k] - p_agents[k]) > message_distance:
                diff = np.linalg.norm(best_relay_points[k] - p_agents[k]) - message_distance
                print(f"Warning: relaying agent {k} could have reached its relay point before the message could have (relay agent moves {diff} further than message)")
            
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



def sample_scenario(K=5, Rcom=1.0, R=5.0, Ra=0.6, sigma=0.1, seed=None):
    rng = np.random.default_rng(seed)

    Rmin = K * Rcom
    Rmax = (K + 4) * Rcom
    dense = R <= (K + 1) * Rcom

    # Transmitting base at origin
    p_tx = np.array([0.0, 0.0])

    # Midpoint for agents initialization
    p0 = np.array([R/2, 0.0])

    # Agent positions around p0
    agent_radiuses = Ra * np.sqrt(rng.uniform(0, 1, size=(K, 1)))    
    agent_angles = rng.uniform(0, 2*np.pi, size=(K, 1))
    p_agents = p0 + agent_radiuses*np.hstack([np.cos(agent_angles), np.sin(agent_angles)])

    # Agent initial orientations
    phi_agents = rng.uniform(0, 2*np.pi, size=K)

    # Receiver base
    p_recv = np.array([R, 0.0])

    # Jammer agent
    p_j = sample_capsule_point(R, Rcom, rng)

    # Jammer displacement (velocity vector)
    # Uniformly distributed in the half plane directed from jammer_pos towards the midpoint between the bases
    pmid = (p_tx + p_recv) / 2
    theta_j_prime = np.arctan2(pmid[1] - p_j[1], pmid[0] - p_j[0])
    theta_j = rng.uniform(theta_j_prime - np.pi/2, theta_j_prime + np.pi/2)
    dp_j = sigma * np.array([np.cos(theta_j), np.sin(theta_j)])  # assume max velocity sigma

    return {
        'R': R,
        'Rmin': Rmin,
        'Rmax': Rmax,
        'dense': dense,
        'p_agents': p_agents,
        'phi_agents': phi_agents,
        'p_recv': p_recv,
        'p0': p0,
        'p_tx': p_tx,
        'Rcom': Rcom,
        'Ra': Ra,
        'p_j': p_j,
        'dp_j': dp_j,
    }


def plot_scenario(sample, directed=False, jammer=False, savepath=None):
    R = sample['R']
    Rmin = sample['Rmin']
    Rmax = sample['Rmax']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
    p0 = sample['p0']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    # Colors and line widths (same as plot_scenario_with_path_colored)
    init_agent_color = "orange"
    receive_color = "blue"
    deliver_color = "green"
    alpha_circle = 0.6
    alpha_init_agent = 0.6
    line_width = 2
    marker_size = 80
    arrow_scale = 10

    fig, ax = plt.subplots(figsize=(6, 6))

    # Transmitting base
    ax.scatter(*p_tx, c=receive_color, marker='s', s=marker_size)#, label='TX base')
    ax.add_artist(plt.Circle(p_tx, Rcom, color=receive_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Midpoint base p0 radius (where agents are initialized)
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color="gray", fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Receiver base
    ax.scatter(*p_recv, c=deliver_color, marker='s', s=marker_size)#, label='RX base')
    ax.add_artist(plt.Circle(p_recv, Rcom, color=deliver_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Plot initial agent positions
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size, zorder=21)

    # Plot initial agent orientations
    if directed:
        phi_agents = sample['phi_agents']
        for i in range(p_agents.shape[0]):
            phi = phi_agents[i]
            p = p_agents[i]
            #ax.arrow(p[0], p[1],
            #             np.sqrt(arrow_scale*sigma)*np.cos(phi), np.sqrt(arrow_scale*sigma)*np.sin(phi), fc='orange', ec='orange', alpha=alpha_init_agent, lw=line_width)
            ax.annotate("", xy=p + arrow_scale*sigma*np.array([np.cos(phi), np.sin(phi)]), xytext=p,    # plot displacement vector (jammer motion direction)
                    arrowprops=dict(arrowstyle="->", color=init_agent_color, lw=line_width))

    # Add Rcom circles around each agent
    for agent_pos in p_agents:
        ax.add_artist(plt.Circle(agent_pos, Rcom, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=0.4, zorder=1))

    # Plot jammer
    if jammer:
        p_j = sample['p_j']
        dp_j = sample['dp_j']
        ax.scatter(p_j[0], p_j[1], c='red', marker='x', s=marker_size+50, label='Jammer')
        ax.annotate("", xy=p_j + arrow_scale*dp_j, xytext=p_j,    # plot displacement vector (jammer motion direction)
                    arrowprops=dict(arrowstyle="->", color='red', lw=line_width))

    # Adjust plot limits
    ax.set_xlim(-0.2*Ra, R+0.2*Ra)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)

    # Annotations
    #ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(p_agents.shape[0]+1)*Rcom:.3f}",
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # Custom legend for scenario
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='TX base', markerfacecolor=receive_color, markersize=10),
        Line2D([0], [0], marker='s', color='w', label='RX base', markerfacecolor=deliver_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Init pos', markerfacecolor=init_agent_color, markersize=10, alpha=alpha_init_agent),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_elements, labels + [le.get_label() for le in legend_elements], loc="upper left")

    ax.set_aspect('equal')
    #ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-0.4 * Ra, R + 0.4 * Ra)
    ax.set_ylim(-1.3 * Ra, 1.3 * Ra)
    plt.title(f"Initial state, K={p_agents.shape[0]}")

    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

def plot_scenario_with_path_colored(sample, dijkstra_result, savepath=None, jammer=False, debug=False):
    R = sample['R']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
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
    line_width = 2
    marker_size = 80

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
    ax.annotate(
        "", xy=ps_k, xytext=p_agents[km],
        arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width, alpha=1.0)
    )

    # Plot dashed line from retrieval point to the Rcom circle of the receiving base
    direction = p_recv - ps_k
    norm = np.linalg.norm(direction)
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

    if debug:  # annotate the agent indices
        for i, pos in enumerate(p_agents):
            ax.text(pos[0], pos[1], str(i), color='black', fontsize=12, ha='center', va='center')

    # Plot the retrieval points
    #ax.scatter(ps_k[0], ps_k[1], c=agent_color, marker='o')


    path = dijkstra_result['path']
    relay_points = dijkstra_result['relay_points']

    # Track if relay points and passive agents are plotted for legend
    relay_point_plotted = False
    passive_agent_plotted = False

    # Plot relay points (destinations) with full opacity, only for relaying agents
    if not debug:
        for idx in dijkstra_result['relaying_agents']:
            relay = relay_points[idx]
            ax.scatter(relay[0], relay[1], c=agent_color, alpha=1.0, marker='o', s=marker_size, zorder=20)
            # Add Rcom circle around relay point
            circle = plt.Circle(relay, Rcom, color=agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
            ax.add_artist(circle)
            relay_point_plotted = True
    else:  # Debug mode: plot all relay points
        for idx in relay_points:
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
            passive_agent_plotted = True

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
    #ax.set_xlim(-0.2*Ra, R+0.2*Ra)
    #ax.set_ylim(-1.1*Ra, 1.1*Ra)
    ax.set_xlim(-0.4 * Ra, R + 0.4 * Ra)
    ax.set_ylim(-1.3 * Ra, 1.3 * Ra)

    # Annotations
    #ax.text(0.02, 0.92, f"R = {R:.3f}\nRmin = {Rmin:.3f}\nRmax = {Rmax:.3f}\nThreshold = {(len(p_agents)+1)*Rcom:.3f}",
    #        transform=ax.transAxes, fontsize=8, verticalalignment='top')

    # TODO plot jammer trajectory, might need to adjust the above since the communication radii will depend on the jammer's position
    if jammer:
        print("Not implemented: jammer plotting")

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
    #ax.grid(True, linestyle='--', alpha=0.6)
    plt.title(
        "Dijkstra Baseline\n"
        f"($K$: {K}; Time steps: {dijkstra_result['delivery_time']}; Value: {dijkstra_result['value']:.2f})"
    )


    if savepath:
        # If folder doesn't exist, create it
        folder = os.path.dirname(savepath)
        if not os.path.exists(folder):
            os.makedirs(folder)

        plt.savefig(savepath, dpi=300)
    plt.show()


def animate_scenario_with_path_colored(sample, dijkstra_result, savepath=None, beamer_gif=False, interval=100):
    
    if savepath and beamer_gif:
        png_folder = savepath[:-4] + "_frames" 
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)

    p_agents = sample['p_agents']
    K = p_agents.shape[0]
    t_start_k = dijkstra_result['agent_start_times']
    t_stop_k = dijkstra_result['agent_stop_times']
    T_D = dijkstra_result['delivery_time']
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
    line_width = 2
    marker_size = 80

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

        ax.set_title(f"Dijkstra Baseline\n ($K$: {K}; Time: {t} / {max(int(np.max(t_stop_k)+1), T_D)} Value: {dijkstra_result['value']:.2f})")
        
        # Save PNG snapshot if requested
        if savepath is not None and beamer_gif:
            fname = os.path.join(png_folder, f"frame_{t:04d}.png")
            plt.savefig(fname, dpi=150)

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
    debug = True

    # Scenario parameters
    K = 4  # something weird with K=3 seed=1  (weird meetup point for retrieving agent and first relay agent)
    for K in range(15,31,5):
        Rcom = 1.0
        R = (K+4)*Rcom
        Ra = 0.6*R

        sigma = 0.1
        beta = 0.99

        directed = False
        jammer = True

        seed=1

        # Cost parameters
        cost_motion = 5
        cost_antenna = 1

        # Saving stuff
        plot_folder = "Plots"
        init_fig_folder = "Initial_states"
        dijk_fig_folder = "Dijkstra_plots"
        anim_folder = "Animations"
        file_name = fr"dijkstra_baseline_K{K}_seed{seed}"

        #save_path = None
        init_save_path = os.path.join(plot_folder, init_fig_folder, file_name+'_init'+'.png') 
        dijkstra_save_path = os.path.join(plot_folder, dijk_fig_folder, file_name+'.png') 
        animation_save_path = os.path.join(anim_folder, file_name+'.gif')
        beamer_gif = False

        sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, sigma=sigma, seed=seed)
        #plot_scenario(sample, directed=directed, jammer=jammer, savepath=init_save_path)
        result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom, sigma=sigma, beta=beta, c_pos=cost_motion, jammer=jammer)    
        #plot_scenario_with_path_colored(sample, result, savepath=dijkstra_save_path, jammer=jammer, debug=debug)
        animate_scenario_with_path_colored(sample, result, savepath=animation_save_path, beamer_gif=beamer_gif)