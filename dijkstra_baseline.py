import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import os
import sys
import numpy.linalg as la 

# Add the parent directory to sys.path to import communication_experimentation
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from communication_experimentation import communication_range


def one_antenna_step(phi_current, phi_final, phi_resolution):
    # Normalize angles to [0, 2π)
    phi_current = phi_current % (2 * np.pi)
    phi_final = phi_final % (2 * np.pi)

    # Compute the shortest angular difference (in range [-π, π))
    delta = (phi_final - phi_current + np.pi) % (2 * np.pi) - np.pi

    # If within one step, go directly to the final angle
    if abs(delta) <= phi_resolution:
        phi_next = phi_final
    else:
        # Step toward phi_final
        step = np.sign(delta) * phi_resolution
        phi_next = (phi_current + step) % (2 * np.pi)

    return phi_next


def jammer_capsule_boundary(R, Rcom, point, cap_height=None, cap_radius=None):
    """Check if point is within jammer capsule boundary"""
    if cap_height is None:
        cap_height = 3 * Rcom
    if cap_radius is None:
        cap_radius = 1.5 * Rcom

    # Rectangle check
    if 0 <= point[0] <= R and -cap_height/2 <= point[1] <= cap_height/2:
        return True

    # Left half circle check
    if la.norm(point - np.array([0, 0])) <= cap_radius:
        return True
    
    # Right half circle check
    if la.norm(point - np.array([R, 0])) <= cap_radius:
        return True
    
    return False

def antenna_gain(theta, C_dir=1.0):
    """Calculate antenna gain |a(0)^H * a(theta)|"""
    # Steering vectors
    a_0 = np.array([1, C_dir])  # a(0)
    a_theta = np.array([1, C_dir * np.exp(1j * np.pi * np.sin(theta))])  # a(theta)
    
    # Antenna gain: |a(0)^H * a(theta)|
    gain = np.abs(np.conj(a_0) @ a_theta)
    return gain


def calculate_sinr_at_receiver(p_tx, p_rec, phi=None, jammer_pos=None):
    """Calculate SINR at receiver position for given phi using the provided equation"""

    if phi:
        # Calculate theta_rec relative to antenna boresight
        angle_to_receiver = np.arctan2(p_rec[1] - p_tx[1], p_rec[0] - p_tx[0])
        theta_rec = np.mod(angle_to_receiver - phi + np.pi, 2*np.pi) - np.pi
        
        # Check if theta_rec is within [-pi/2, pi/2]
        if abs(theta_rec) > np.pi/2:
            return 0.0, theta_rec
        
        # Calculate antenna gain
        antenna_gain_value = antenna_gain(theta_rec, C_dir=1.0)
    else:
        antenna_gain_value = 1.0
        theta_rec = None
        
    # Calculate distances
    dist_tr_squared = la.norm(p_rec - p_tx)**2  # ||p_r - p_t||^2
    
    if jammer_pos is not None:
        C_jam = 3.0
        dist_jr_squared = la.norm(p_rec - jammer_pos)**2  # ||p_r - p_j||^2
        denominator = dist_tr_squared * (1 + C_jam / dist_jr_squared)
    else:
        denominator = dist_tr_squared  
    
    sinr = antenna_gain_value / denominator
    
    return sinr, theta_rec


def compute_retrieval_point(pk, Rcom, p_tx=np.array([0.0, 0.0])):
    v = pk - p_tx 
    norm_pk = la.norm(v)
    if norm_pk > Rcom:
        return (Rcom / norm_pk) * v + p_tx
    else:
        return pk


def compute_handover_point(p_from, p_to, Rcom):
    v = p_to - p_from
    norm_v = la.norm(v)
    if norm_v > Rcom:
        return p_to - Rcom * (v / norm_v)
    else:
        return p_from


def compute_relay_point(pj, ps_k, pk, pr, j_k, Rcom, clustering=False, Iright=None, p_rk=None, epsilon=1e-3):
    """
    Compute the relay point for agent j_k (index in sorted projection order, starting from 1):
    - Project pj onto the line from ps_k to pr to get bar_pj (pmeet_j).
    - If agent can reach bar_pj in time, return bar_pj.
    - Otherwise, return q_j as described in the prompt.
    """

    # If clustering is enabled and j_k is in Iright, try to move to clustered position
    if clustering and Iright is not None and p_rk is not None:
        v = ps_k - pr
        norm_v = la.norm(v)
        v = v / norm_v
        if j_k in Iright:
            idx = Iright.index(j_k)
            target = p_rk + (len(Iright)-idx)*epsilon*v
            c = la.norm(ps_k - pk)
            d = la.norm(p_rk - ps_k)
            if la.norm(p_rk - pj) <= c + max(0, d - (len(Iright)+1)*Rcom):
                return target
            else:
                return pj

    # Direction from ps_k to pr
    u = pr - ps_k
    norm_u = la.norm(u)
    u = u / norm_u

    # Projection of pj onto the line (pmeet_j)
    bar_pj = ps_k + np.dot(pj - ps_k, u) * u

    a = la.norm(bar_pj - pj)
    c = la.norm(ps_k - pk)
    d = la.norm(bar_pj - ps_k)
    #a = la.norm(bar_pj - ps_k) - j_k*Rcom

    # Condition for using the projection point (agent makes it to its projection before the message does)
    if a <= c + max(0, d - j_k*Rcom):  # lambda <= 0
        return bar_pj

    # Agent doesn't make it to its projection before the message does
    # v = unit normal from line to pj
    diff = pj - bar_pj
    norm_diff = la.norm(diff)
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


def compute_budget(K, R, sigma, beta):
    """
    Compute the budget(w) as described in the prompt.

    Args:
        K (int): Number of agents
        R (float): Distance between transmitter and receiver
        sigma (float): Displacement size
        beta (float): Discount factor
        w (int): 1 if message delivered, 0 otherwise

    Returns:
        float: The computed budget(w)
    """
    Rcom = 1.0

    # Compute T_sharp
    T_sharp = int(np.floor(1.1 * R + 2 * Rcom) / sigma + K)

    # Compute D_k for all agents
    D = np.zeros(K)
    for k in range(1, K+1):
        if k == 1:
            D[k-1] = 1.1 * R + 2 * Rcom
        else:
            D[k-1] = 0.1 * R + (K - k + 1) * Rcom

    # Compute t_start_k and t_stop_k for all agents
    t_start = np.zeros(K, dtype=int)
    t_stop = np.zeros(K, dtype=int)
    for k in range(1, K+1):
        if k == 1:
            t_start[k-1] = 0
            t_stop[k-1] = int(np.ceil(D[0] / sigma))
        else:
            t_start[k-1] = int(np.floor((D[0] - D[k-1]) / sigma)) + (k - 1)
            t_stop[k-1] = T_sharp - (k + 1)

    # Compute budget
    budget = 0.0
    for t in range(T_sharp):
        beta_t = beta ** t
        active_agents = sum(
            t_start[k] <= t < t_stop[k] for k in range(K)
        )
        budget += beta_t * sigma**2 * active_agents
    budget = budget / (beta ** T_sharp)

    return budget


def dijkstra_baseline(
    p_agents, 
    p_tx, 
    p_recv, 
    Rcom=1.0, 
    sigma=0.2, 
    beta=0.99, 
    c_pos=0.1, 
    c_phi=0,
    phi_agents=False, 
    jammer_info=None, 
    debug=False, 
    clustering=False, 
    epsilon=1e-3,
    minimize_distance=False
):
    K = p_agents.shape[0]
    R = la.norm(p_recv - p_tx)
    I = list(range(K))
    best_path = None
    best_distance = np.inf
    best_k = None
    best_graph = None
    best_relay_points = None
    best_relaying_agents = None
    best_first_relay_points = None
    directed_transmission = (phi_agents is not None)
    jammer_on = (jammer_info is not None)

    # Store all candidate paths and relay points for later agent movement computation
    candidate_paths = []
    candidate_relay_points = []
    candidate_graphs = []
    candidate_k = []

    # Dijkstra iterations 
    n_iter = 2
    for iter_idx in range(n_iter):
        if iter_idx == 0:
            # Use all agents for the first iteration
            agent_indices = I
        else:
            # Use only active agents from the previous best path
            agent_indices = [best_k] + best_relaying_agents

        # For all possible retriving agents
        for k in agent_indices:
            pk = p_agents[k]
            ps_k = compute_retrieval_point(pk, Rcom, p_tx)
            pr = p_recv

            # Build graph G_k
            Gk = nx.DiGraph()
            node_map = {}
            Gk.add_node(k, pos=pk)
            Gk.add_node('s_k', pos=ps_k)
            Gk.add_node('r', pos=pr)

            # Direction vector from retrieval point to receiver
            direction = pr - ps_k
            norm_dir = la.norm(direction)
            direction = direction / norm_dir

            # Project all other agents onto the line and sort by increasing distance from ps_k to projection point
            agent_projections = []
            for j in agent_indices:
                if j == k:
                    continue
                pj = p_agents[j]
                u = direction
                proj_pj = ps_k + np.dot(pj - ps_k, u) * u
                dist = la.norm(proj_pj - ps_k)
                if 0 <= np.dot(proj_pj - ps_k, pr - ps_k):  # only consider projections in the direction of pr from ps_k
                    agent_projections.append((j, proj_pj, dist))
            sorted_agents = [j for j, proj, dist in sorted(agent_projections, key=lambda x: x[2])]

            # Add relay agents and compute relay points
            for idx, j in enumerate(sorted_agents):
                pj = p_agents[j]
                relay_j = compute_relay_point(pj, ps_k, pk, pr, j_k=idx+1, Rcom=Rcom)  # j_k=idx+1 more conservative (slower but higher value due to less motion) than j_k=idx
                node_map[j] = relay_j
                Gk.add_node(j, pos=relay_j)

            # Edge set
            Gk.add_edge(k, 's_k', weight=la.norm(ps_k - pk))
            Gk.add_edge('s_k', 'r', weight=max(0, la.norm(pr - ps_k) - Rcom))
            for j, relay_j in node_map.items():
                if j != k:
                    w = max(0, la.norm(relay_j - ps_k) - Rcom)
                    Gk.add_edge('s_k', j, weight=w)
                    Gk.add_edge(j, 'r', weight=max(0, la.norm(pr - relay_j) - Rcom))
            for j, relay_j in node_map.items():
                if j != k:
                    for l, relay_l in node_map.items():
                        if j != l and l != k:
                            Gk.add_edge(j, l, weight=max(0, la.norm(relay_l - relay_j) - Rcom))

            # Run Dijkstra from k to r
            try:
                path = nx.dijkstra_path(Gk, source=k, target='r', weight='weight')
                total_dist = sum(Gk.edges[path[i], path[i+1]]['weight'] for i in range(len(path)-1))
                candidate_paths.append((path, total_dist))
                candidate_relay_points.append(node_map)
                candidate_graphs.append(Gk)
                candidate_k.append(k)

                # Check if path is valid (don't allow relay points within Rcom of TX base)
                valid_path = True
                for node in path:
                    if isinstance(node, int) and node in node_map:
                        relay_point = node_map[node]
                        if la.norm(relay_point - p_tx) < Rcom:
                            valid_path = False
                            break

                if total_dist <= best_distance and valid_path:
                    best_distance = total_dist
                    best_path = path
                    best_k = k
                    best_graph = Gk
                    best_relay_points = node_map

                    # Save passive relay points from first iteration
                    if iter_idx == 0:  
                        best_first_relay_points = node_map.copy()
            except nx.NetworkXNoPath:
                continue

        # Prepare for next iteration: extract relaying agents from best path
        best_relaying_agents = [n for n in best_path if isinstance(n, int) and n != best_k]

    # Agent logging 
    all_active = set([best_k] + best_relaying_agents)
    passive_agents = [idx for idx in range(K) if idx not in all_active]  
            
    # Passive agent relay points (for evaluation purposes)
    passive_relay_points = {}
    for j in range(K):  # all agents from first iteration
        if j in passive_agents and j in list(best_first_relay_points.keys()):
            passive_relay_points[j] = best_first_relay_points[j]
        elif j not in all_active:  # agents that were to the left of transmitting base and not retriever
            passive_relay_points[j] = p_agents[j]

    # Remove passive agents from best_relay_points
    for j in passive_agents:
        if j in best_relay_points:
            del best_relay_points[j]

    # Clustering extension: spread out relaying agents after second Dijkstra
    if clustering:
        # 1. Find p_{r_k}
        pk = p_agents[best_k]
        ps_k = compute_retrieval_point(pk, Rcom, p_tx)
        pr = p_recv
        v = (ps_k - pr) / la.norm(ps_k - pr)
        pr_k = pr + Rcom * v

        # Identify agents in Iright (projected further than p_{r_k})
        Iright = []
        for idx in range(K):
            pj = p_agents[idx]
            u = (pr - ps_k) / la.norm(pr - ps_k)
            proj_pj = ps_k + np.dot(pj - ps_k, u) * u
            if np.dot(proj_pj - pr_k, u) >= 0 and idx != best_k:
                Iright.append(idx)
        # Sort Iright by distance from p_{r_k}
        Iright = sorted(Iright, key=lambda j: la.norm(p_agents[j] - pr_k))

        # For each agent in Iright, try to move to p_{r_k} + (|Iright|-j)*epsilon*v, ignore the ones who can't make it before the message
        right_agents = {}
        for i, j in enumerate(Iright):
                pj = p_agents[j]
                target = pr_k + (len(Iright)-i)*epsilon*v
                c = la.norm(ps_k - pk)
                d = la.norm(target - ps_k)
                if la.norm(target - pj) <= c + max(0, d - (len(best_relaying_agents)+1+i)*Rcom):
                    right_agents[j] = target 

        # Include right agents that made it to their relaying point into the relay agent set
        relay_points_clustered = best_relay_points.copy()
        for j, pos in right_agents.items():
            relay_points_clustered[j] = pos

        # Agents considered for clustering
        cluster_agents = list(relay_points_clustered.keys())  # already sorted
        # cluster_agents = sorted(cluster_agents, key=lambda j: la.norm(relay_points_clustered[j] - ps_k))

        clusters_prev = None
        previous_sorted_clusters = []
        while True:
            #print(clusters_prev)
            # 2. Compute movement budgets for all relay agents
            movement_budget = {}
            for j in cluster_agents:
                hat_pj = relay_points_clustered[j]
                c = la.norm(ps_k - pk)
                d = la.norm(hat_pj - ps_k)
                j_k = cluster_agents.index(j) + 1  # order in relaying chain
                remaining_budget = c + max(0, d - j_k*Rcom) - la.norm(hat_pj - p_agents[j])
                movement_budget[j] = remaining_budget

                # Sometimes the agents are not at the l_{ps_k,pr_k} line even with positive budget, these should be seen as fixed
                proj_point_j = ps_k + np.dot(hat_pj - ps_k, v)*v
                dist_to_line = la.norm(hat_pj - proj_point_j)
                if dist_to_line > 1e-3 and movement_budget[j] > 0:
                    movement_budget[j] = 0.0

            # 3. Partition into fixed and movable agents
            If = [j for j, b in movement_budget.items() if np.isclose(b, 0) or b < 0]  # note that budget might be negative due to extra Dijkstra run where previous agent relay positions have been taken into account 

            # 4. Cluster formation (using communication graph)
            cluster_graph = nx.Graph()
            for j in cluster_agents:
                cluster_graph.add_node(j)
            for i, j in enumerate(cluster_agents):
                for l in cluster_agents[i+1:]:
                    if la.norm(relay_points_clustered[j] - relay_points_clustered[l]) <= Rcom:
                        cluster_graph.add_edge(j, l)
            clusters = [set(c) for c in nx.connected_components(cluster_graph)]  # uses BFS

            # 5. Redistribute agents along the line for each cluster
            v = (pr-ps_k)/la.norm(pr-ps_k)
            left_limit = ps_k + Rcom*v

            # Compute right_limit as the minimum of pr and the second clusters leftmost agent's position
            if len(clusters) > 1:
                second_leftmost_agent = min(clusters[1], key=lambda j: np.dot(relay_points_clustered[j], v))
                
                leftmost_proj = ps_k + np.dot(relay_points_clustered[second_leftmost_agent] - ps_k, v)*v  # make sure lies on line
                right_limit = leftmost_proj
                #right_limit = relay_points_clustered[second_leftmost_agent] - Rcom * v
                if np.dot(right_limit - ps_k, v) > np.dot(pr - ps_k, v):
                    right_limit = pr - Rcom * v
            else:
                right_limit = pr - Rcom * v

            n_clusters = len(clusters)
            for cluster_count, cluster in enumerate(clusters):
                n_cluster = len(cluster)  # number of agents in the cluster
                if n_cluster > 1:  
                    if not any(j in If for j in cluster):  # no fixed agents
                        if n_cluster % 2 == 0:  # even number of agents
                            centroid = np.mean([relay_points_clustered[j] for j in cluster], axis=0)
                            proj_dists = [np.dot(relay_points_clustered[j] - centroid, v) for j in cluster]
                            sorted_cluster = [j for _, j in sorted(zip(proj_dists, cluster))]
                            for idx, j in enumerate(sorted_cluster):
                                offset = (idx - (n_cluster-1)/2) * Rcom
                                intended_target = centroid + offset * v
                            
                                # Respect right_limit
                                if np.dot(intended_target - ps_k, v) > np.dot(right_limit - ps_k, v):
                                    intended_target = right_limit
                                # Respect left_limit
                                if np.dot(intended_target - ps_k, v) < np.dot(left_limit - ps_k, v):  # end up here when previous agent is a fixed agent with Rcom radius sticking to the right of current agent's relay position
                                    #intended_target = left_limit
                                    intended_target = relay_points_clustered[j]  # stand still

                                # Respect movement budget
                                disp_vec = intended_target - relay_points_clustered[j]
                                disp_j = la.norm(disp_vec)
                                if movement_budget[j] > 0:
                                    if disp_j < movement_budget[j]:
                                        relay_points_clustered[j] = intended_target
                                    else:
                                        relay_points_clustered[j] = relay_points_clustered[j] + (movement_budget[j] / disp_j) * disp_vec if disp_j > 1e-12 else relay_points_clustered[j]
                                else:
                                    relay_points_clustered[j] = relay_points_clustered[j]  # no movement

                        else:  # odd number of agents
                            mid = n_cluster // 2 + 1
                            sorted_cluster = sorted(cluster, key=lambda j: la.norm(relay_points_clustered[j] - ps_k))
                            
                            # Place agents to the right of the middle
                            for i in range(mid, n_cluster):
                                prev_j = sorted_cluster[i-1]
                                j = sorted_cluster[i]

                                # Desired target: Rcom away from previous agent, along +v
                                target = relay_points_clustered[prev_j] + Rcom * v

                                # Respect right_limit
                                if np.dot(target - ps_k, v) > np.dot(right_limit - ps_k, v):
                                    target = right_limit
                                # Respect movement budget
                                disp_j = la.norm(target - relay_points_clustered[j])
                                if movement_budget[j] > 0:
                                    if disp_j < movement_budget[j]:
                                        relay_points_clustered[j] = target
                                    else:
                                        relay_points_clustered[j] = relay_points_clustered[j] + movement_budget[j] * v
                            # Place agents to the left of the middle
                            for i in range(mid-1, 0, -1):
                                next_j = sorted_cluster[i]
                                j = sorted_cluster[i-1]

                                # Desired target: Rcom away from previous agent, along +v
                                target = relay_points_clustered[next_j] - Rcom * v

                                # Respect left_limit
                                if np.dot(target - ps_k, v) < np.dot(left_limit - ps_k, v):
                                    target = left_limit
                                # Respect movement budget
                                disp_j = la.norm(target - relay_points_clustered[j])
                                if movement_budget[j] > 0:
                                    if disp_j < movement_budget[j]:
                                        relay_points_clustered[j] = target
                                    else:
                                        relay_points_clustered[j] = relay_points_clustered[j] - movement_budget[j] * v
                            
                    else:  # cluster has a fixed agent, only move the others if they have budget
                        fixed_agents = [j for j in cluster if j in If]

                        # Sort fixed agents by projection distance from ps_k
                        fixed_agents = sorted(fixed_agents, key=lambda j: np.dot(relay_points_clustered[j], v))

                        # Indices of boundary fixed agents
                        if len(fixed_agents) == 1:
                            j_left_fixed = fixed_agents[0]
                            j_right_fixed = fixed_agents[0]
                        else:
                            j_left_fixed = fixed_agents[0]
                            j_right_fixed = fixed_agents[-1]
                            
                        # Sort cluster agents by relay order  
                        proj_dists = [np.dot(relay_points_clustered[j] - ps_k, v) for j in cluster]
                        sorted_cluster = [j for _, j in sorted(zip(proj_dists, cluster))]
                        #sorted_cluster = [j for j in relay_points_clustered if j in cluster]
                        idx_left_fixed = sorted_cluster.index(j_left_fixed)
                        idx_right_fixed = sorted_cluster.index(j_right_fixed)

                        # Rearrange agents to the left of the leftmost fixed agent
                        for i in range(idx_left_fixed-1, -1, -1):  # from closest to fixed, moving outward
                            j = sorted_cluster[i]
                            j_fixed = j_left_fixed
                            p_fixed = relay_points_clustered[j_fixed]
                            # The previous agent is either the fixed agent or the last moved agent
                            if i == idx_left_fixed-1:
                                next_pos = p_fixed
                                # Move away from fixed agent as far as possible without losing contact
                                next_proj = ps_k + np.dot(next_pos - ps_k, v)*v
                                next_proj_dist = la.norm(next_pos - next_proj)
                                if next_proj_dist < Rcom:
                                    t = np.sqrt(Rcom**2 - next_proj_dist**2)
                                else:
                                    raise RuntimeError("Clustering error: fixed agent too far to maintain contact")
                                
                                target = next_proj - t * v
                            else:
                                next_pos = relay_points_clustered[sorted_cluster[i+1]]
                                # Desired target: Rcom away from previous agent, along -v
                                target = next_pos - Rcom * v
                            # Respect left_limit
                            if np.dot(target - ps_k, v) < np.dot(left_limit - ps_k, v):
                                # Make sure limit is on the line, stand still if not
                                limit_proj = ps_k + np.dot(left_limit - ps_k, v)*v
                                dist_to_line = la.norm(limit_proj - left_limit)
                                if dist_to_line > 1e-3 :
                                    target = relay_points_clustered[j]  # stand still
                                else:
                                    target = left_limit  

                            # Respect movement budget
                            disp_j = la.norm(target - relay_points_clustered[j])
                            if movement_budget[j] > 0:
                                if disp_j < movement_budget[j]:
                                    relay_points_clustered[j] = target
                                else:
                                    relay_points_clustered[j] = relay_points_clustered[j] - movement_budget[j] * v
                        
                        # Rearrange agents to the right of the rightmost fixed agent
                        for i in range(idx_right_fixed+1, n_cluster):
                            j = sorted_cluster[i]
                            j_fixed = j_right_fixed
                            p_fixed = relay_points_clustered[j_fixed]
                            # The previous agent is either the fixed agent or the last moved agent
                            if i == idx_right_fixed+1:
                                # Move away from fixed agent as far as possible without losing contact
                                prev_proj = ps_k + np.dot(p_fixed - ps_k, v)*v
                                prev_proj_dist = la.norm(p_fixed - prev_proj)
                                if prev_proj_dist < Rcom:
                                    t = np.sqrt(Rcom**2 - prev_proj_dist**2)
                                else:  
                                    raise RuntimeError("Clustering error: fixed agent too far to maintain contact")
                                
                                target = prev_proj + t * v
                            else:
                                # Desired target: Rcom away from previous (nonfixed) agent, along +v
                                prev_pos = relay_points_clustered[sorted_cluster[i-1]]
                                target = prev_pos + Rcom * v
                            # Respect right_limit
                            if np.dot(target - ps_k, v) > np.dot(right_limit - ps_k, v):
                                # Make sure limit is on the line, stand still if not
                                limit_proj = ps_k + np.dot(right_limit - ps_k, v)*v
                                dist_to_line = la.norm(limit_proj - right_limit)
                                if dist_to_line > 1e-3 :
                                    target = relay_points_clustered[j]  # stand still
                                else:
                                    target = right_limit 
                            # Respect movement budget
                            disp_j = la.norm(target - relay_points_clustered[j])
                            if movement_budget[j] > 0:
                                if disp_j < movement_budget[j]:
                                    relay_points_clustered[j] = target
                                else:
                                    relay_points_clustered[j] = relay_points_clustered[j] + movement_budget[j] * v

                # Move the left limit to the rightmost agent's projected position in current cluster
                rightmost_j = max(cluster, key=lambda j: np.dot(relay_points_clustered[j], v))

                rightmost_proj = ps_k + np.dot(relay_points_clustered[rightmost_j] - ps_k, v)*v
                left_limit = rightmost_proj
                #left_limit = relay_points_clustered[rightmost_j] + Rcom * v

                # Move the right limit to the leftmost agent position in the next cluster - Rcom
                if cluster_count + 2 < n_clusters - 1:  
                    next_cluster = clusters[cluster_count + 2]
                    leftmost_j_next = min(next_cluster, key=lambda j: np.dot(relay_points_clustered[j], v))
                    
                    leftmost_proj = ps_k + np.dot(relay_points_clustered[leftmost_j_next] - ps_k, v)*v  # make sure lies on line
                    right_limit = leftmost_proj

                    #right_limit = relay_points_clustered[leftmost_j_next] - Rcom * v
                    if np.dot(right_limit - ps_k, v) > np.dot(pr - ps_k, v):
                        right_limit = pr - Rcom * v
                else:
                    right_limit = pr - Rcom * v


            # 6. Break if no new clusters have been formed
            if clusters_prev is not None:
                # Compare clusters by their sets of members
                clusters_prev_sorted = sorted([sorted(list(c)) for c in clusters_prev])
                clusters_sorted = sorted([sorted(list(c)) for c in clusters])
                if clusters_sorted == clusters_prev_sorted:
                    break  # No change, clustering is done
            clusters_prev = [set(c) for c in clusters]
            previous_sorted_clusters.append(clusters_prev)

            # If we've stumbled into a cycle, break
            if clusters_prev in previous_sorted_clusters[:-1]:
                break
        # END WHILE clustering
    
        # Rebuild best_graph with new relay points
        Gk = nx.DiGraph()
        Gk.add_node(best_k, pos=p_agents[best_k])
        Gk.add_node('s_k', pos=ps_k)
        Gk.add_node('r', pos=pr)
        for j, pos in relay_points_clustered.items():
            Gk.add_node(j, pos=pos)
        Gk.add_edge(best_k, 's_k', weight=la.norm(ps_k - p_agents[best_k]))
        Gk.add_edge('s_k', 'r', weight=max(0, la.norm(pr - ps_k) - Rcom))
        for j, pos in relay_points_clustered.items():
            if j != best_k:
                w = max(0, la.norm(pos - ps_k) - Rcom)
                Gk.add_edge('s_k', j, weight=w)
                Gk.add_edge(j, 'r', weight=max(0, la.norm(pr - pos) - Rcom))
        for j, pos_j in relay_points_clustered.items():
            if j != best_k:
                for l, pos_l in relay_points_clustered.items():
                    if j != l and l != best_k:
                        Gk.add_edge(j, l, weight=max(0, la.norm(pos_l - pos_j) - Rcom))
        best_graph = Gk

        # Compute the best path
        try:
            best_path = nx.dijkstra_path(best_graph, source=best_k, target='r', weight='weight')
            best_distance = sum(best_graph.edges[best_path[i], best_path[i+1]]['weight'] for i in range(len(best_path)-1))
        
            best_relay_points = {}
            for n in best_path:
                if isinstance(n, int) and n != best_k:
                    best_relay_points[n] = relay_points_clustered[n]

        except nx.NetworkXNoPath:
            raise ValueError("No path found in clustered graph, something went wrong")

        # Adjust for previously passive agents that are now included after clustering
        best_relaying_agents = list(best_relay_points.keys())
        all_active = [best_k] + best_relaying_agents
        passive_agents = [idx for idx in range(K) if idx not in all_active]
        passive_relay_points = {}
        for j in passive_agents:
            if j in passive_agents and j in list(best_first_relay_points.keys()):
                passive_relay_points[j] = best_first_relay_points[j]
            elif j not in all_active:  # agents that were to the left of transmitting base and not retriever
                passive_relay_points[j] = p_agents[j]

    # END IF clustering

    # Compute distances, times, etc for best path for the current scenario (isotropic/directed transmission & nonjammed/jammed)
    # Initialize timing arrays
    # Isotropic nonjammed scenario, use this solution for the other scenarios (the times and distances will change for the other scenarios)
    k_D = len(best_relaying_agents)
    T_M = int(np.ceil(best_distance / sigma))
    T_D = T_M + k_D + 2

    D_k = np.zeros(K)  # agent distances
    t_start_k = np.zeros(K, dtype=int)   # agent start movement times
    t_stop_k = -1*np.ones(K, dtype=int)  # agent stop movement times, -1 if not moving
    handover_points = {}  # store handover points for each relay (+receiving) agent (points at which the agents stop moving)

    # Distance and stop time for retrieving agent
    ps_k = best_graph.nodes['s_k']['pos'] 
    D_k[best_k] = la.norm(- p_agents[best_k]) 
    if k_D > 0:
        first_relay = best_relaying_agents[0]
        p_to=best_relay_points[first_relay]
    else:
        p_to=p_recv

    handover_point_best_k = compute_handover_point(
            p_from=ps_k,
            p_to=p_to,
            Rcom=Rcom
        )

    D_ret_hand = la.norm(handover_point_best_k - ps_k)
    D_k[best_k] += D_ret_hand
    t_stop_k[best_k] = int(np.ceil(la.norm(ps_k - p_agents[best_k]) / sigma)) + 1 + int(np.ceil(D_ret_hand / sigma))
    handover_points[best_k] = handover_point_best_k

    # Distances and times for relaying agents  (recall that everything continues being isotropic)
    for idx, k in enumerate(best_relaying_agents):
        relay_point = best_relay_points[k]

        if idx < k_D - 1:
            k_next = best_relaying_agents[idx+1]
            p_to = best_relay_points[k_next]
        else:
            p_to = p_recv

        handover_point_k = compute_handover_point(
            p_from=relay_point,
            p_to=p_to,
            Rcom=Rcom
        )
        handover_points[k] = handover_point_k

        D_to_rel = la.norm(relay_point - p_agents[k])
        D_to_hand = la.norm(handover_point_k - relay_point)
        D_k[k] = D_to_rel + D_to_hand

        # Start time: must arrive at relay point before the preceding agent arrives
        t_rel_dur = int(np.ceil(D_to_rel / sigma))
        if idx == 0:
            # First relay after retrieving agent
            t_start_k[k] = t_stop_k[best_k] - t_rel_dur + 1
        else:
            prev_relay = best_relaying_agents[idx-1]
            prev_arrival = t_stop_k[prev_relay]
            t_start_k[k] = prev_arrival - t_rel_dur + 1

        # Stop time
        t_hand_dur = int(np.ceil(D_to_hand / sigma))
        t_stop_k[k] = t_start_k[k] + t_rel_dur + 1 + t_hand_dur
    
            
    # Sanity check, no active agent has reached its relay point before the message could have (excluding air distance for the message)
    if debug:
        if not directed_transmission and not jammer_on:
            for idx, k in enumerate(best_relaying_agents):
                message_distance = la.norm(ps_k - p_agents[best_k]) + \
                    max(0, la.norm(best_relay_points[best_relaying_agents[0]] - ps_k) - Rcom)
                if idx > 0:
                    message_distance += sum(
                        max(0, la.norm(best_relay_points[best_relaying_agents[i+1]] - best_relay_points[best_relaying_agents[i]]) - Rcom) for i in range(idx)  # distance to next relay agent
                    )

                if la.norm(best_relay_points[k] - p_agents[k]) > message_distance:
                    diff = la.norm(best_relay_points[k] - p_agents[k]) - message_distance
                    print(f"Warning: relaying agent {k} could have reached its relay point before the message could have (relay agent moves {diff} further than message)")
        else:
            Warning("Sanity check for agent relay times not implemented for scenarios with orientations or jammers")

        
    # Non-isotropic scenarios
    antenna_directions_all = {}  # store antenna directions at each time step for each active agent (while they are active)  
    t_pickup_all = {}  # store pickup times for each active agent
    p_jammer_current = None
    v_jammer_current = None
    p_jammer_all = None
    v_jammer_all = None
    phi_costs = np.zeros(K)  # store antenna steering costs for each agent
    if directed_transmission or jammer_on:  # Directed transmission or jammer  

        if directed_transmission:
            phi_max_steps = 8
            phi_resolution = np.pi / phi_max_steps
            n_possible_phi = int((2*np.pi) / phi_resolution)
        else:
            phi_max_steps = 0

        if jammer_on:
            p_jammer_start = jammer_info['p_jammer']
            v_jammer_start = jammer_info['v_jammer']
            p_jammer_all = {}
            p_jammer_all[0] = p_jammer_start # collect all jammer positions at each time point
            v_jammer_all = {}
            v_jammer_all[0] = v_jammer_start  # collect all jammer velocities at each time point

        # Process each agent in the relay chain
        for idx in range(k_D + 1):  # +1 to include retrieving agent
            has_received_message = False

            # Determine pickup point
            if idx == 0:
                k_current = best_k
                if jammer_on:
                    p_pickup = p_tx
                else:
                    p_pickup = ps_k
                t_start = 0  
            else:
                k_current = best_relaying_agents[idx - 1]
                p_pickup = best_relay_points[k_current]
                t_start = t_start_k[k_current]

            # Determine target
            if idx == k_D:  # last relay agent
                p_target = p_recv  # last relay sends to receiver base
                p_next_relay = p_recv
            else:
                k_next = best_relaying_agents[idx] if idx == 0 else best_relaying_agents[idx]
                p_next_relay = best_relay_points[k_next]

                # Propagate next agent until t_start if needed
                t_next_ahead = t_start - t_start_k[k_next]  
                if t_next_ahead > 0:
                    p_next_agent = p_agents[k_next].copy()
                    for _ in range(t_next_ahead):
                        p_next_direction = p_next_relay - p_agents[k_next]
                        p_next_distance = la.norm(p_next_direction)
                        if p_next_distance >= sigma:
                            p_next_agent = p_next_agent + sigma * (p_next_direction / p_next_distance)

                    p_target = p_next_agent.copy()
                else:
                    p_target = p_agents[k_next].copy()
                    p_next_agent = p_agents[k_next].copy()

            # Retrieve starting parameters for current agent
            t_current = t_start
            p_start = p_agents[k_current]

            # Compute time and distance for retrieving agent to reach pickup point from TX base
            D_pickup = la.norm(p_pickup - p_start)
            t_pickup_dur = int(np.ceil(D_pickup / sigma)) + 1  # +1 for pickup time
            t_pickup = t_start + t_pickup_dur

            # Compute maximum time limits
            p_handover_max = compute_handover_point(p_pickup, p_next_relay, Rcom)
            D_hand_max = la.norm(p_handover_max - p_pickup)
            t_move_dur_max = int(np.ceil(D_hand_max / sigma))
            t_dur_max_temp = t_pickup_dur + 1 + t_move_dur_max  # max total time this agent is active
            t_max = t_start + t_dur_max_temp
            t_max_w_antenna = t_max + phi_max_steps  # allow extra time for antenna steering
                
            # Initialize simulation variables
            p_current = p_start.copy()
            handover_position = None
            handover_complete = False
            
            # Store antenna direction at each time point the current agent "active" meaning has started moving but have not relayed the message yet
            if directed_transmission:
                phi_start = phi_agents[k_current]
                phi_current = phi_start
                phi_possible = [(phi_start + m * phi_resolution) % (2 * np.pi) for m in range(n_possible_phi)]
                antenna_directions = [phi_start]
            else:
                phi_current = None

            # Initialize jammer position tracking
            if jammer_on:
                p_jammer_current = p_jammer_all[t_current].copy()
                v_jammer_current = v_jammer_all[t_current].copy()
            
            # While loop to find when communication becomes possible
            while t_current <= t_max_w_antenna and not handover_complete:

                # Check if current agent has received the message from previous entity (either transmitting base or preceding agent)
                if (t_current > 0) and not has_received_message:
                    if jammer_on and idx == 0:
                        sinr_pickup, _ = calculate_sinr_at_receiver(
                            p_pickup, p_current, phi_current, jammer_pos=p_jammer_current
                        )
                        if sinr_pickup >= 1.0:
                            within_pickup_range = True
                            if idx == 0:  # new pickup for retrieving agent
                                ps_k = p_current.copy()
                        else:
                            within_pickup_range = False
                    else:
                        within_pickup_range = (la.norm(p_pickup - p_current) < sigma)
                    if within_pickup_range:
                        has_received_message = True
                        t_pickup_all[k_current] = t_current

                # Check if communication is possible at current position and antenna direction between the current agent and the next agent
                sinr, _ = calculate_sinr_at_receiver(
                    p_current, p_target, phi_current, jammer_pos=p_jammer_current
                )
                
                # Communication is possible if SINR > threshold and within range
                communication_possible = (sinr >= 1.0)  # hardcoded SNR threshold = 1
                
                if communication_possible and has_received_message:  # must be after pickup time
                    handover_position = p_current.copy()
                    handover_complete = True
                    break
                
                # Step time forward  
                t_current += 1
                    
                if directed_transmission:
                    # Compute antenna orientation towards target (either next agent or receiving base)
                    direction_to_target = p_target - p_current
                    target_angle = np.mod(np.arctan2(direction_to_target[1], direction_to_target[0]), 2*np.pi)
                    
                    # Find closest allowed antenna orientation 
                    angle_diffs = [abs(target_angle - angle) for angle in phi_possible]
                    # Handle wraparound
                    angle_diffs = [min(diff, 2*np.pi - diff) for diff in angle_diffs]
                    best_m = np.argmin(angle_diffs)
                    phi_target = phi_possible[best_m]
                    
                    # Update antenna direction (one step towards target if needed)
                    phi_next = one_antenna_step(phi_current, phi_target, phi_resolution)
                    phi_current = phi_next
                    
                    # Store antenna direction for this time step
                    antenna_directions.append(phi_current)
                
                # Update position (move towards target)
                if jammer_on:
                    within_range = False  # ignore range constraints when jammers are present
                else:
                    within_range = la.norm(p_next_relay - p_current) <= Rcom  # do not move agent if it is within range of next relay point (might just need to redirect antenna)
                admissible_move = (t_current <= t_max) and not (within_range and has_received_message)
                    
                if admissible_move:
                    # Move transmitting agent towards waypoint (either receiving base or relay point)
                    if has_received_message:
                        p_move_target = p_recv if idx == k_D else p_next_relay
                    else:
                        p_move_target = p_pickup

                    direction_to_move = p_move_target - p_current
                    distance_to_move = la.norm(direction_to_move)
                    
                    if distance_to_move >= sigma:
                        # Move one step towards target
                        p_current = p_current + sigma * (direction_to_move / distance_to_move)

                # Move the potential next agent towards its relay point (if applicable --- not for agent relaying to receiving base)
                if (idx < k_D) and (t_start_k[k_next] <= t_current) and (t_current < t_stop_k[k_next]):
                    p_next_direction = p_next_relay - p_next_agent
                    p_next_distance = la.norm(p_next_direction)
                    
                    if p_next_distance > sigma:
                        # Move one step towards target
                        p_next_agent = p_next_agent + sigma * (p_next_direction / p_next_distance)
                    else:
                        # Reached target position
                        p_next_agent = p_next_relay

                    p_target = p_next_agent

                # Move jammer
                if jammer_on:
                    if not jammer_capsule_boundary(R, Rcom, point=p_jammer_current):
                        v_jammer_current = -v_jammer_current
                    p_jammer_current = p_jammer_current + v_jammer_current
                    p_jammer_all[t_current] = p_jammer_current.copy()
                    v_jammer_all[t_current] = v_jammer_current.copy()


            # END WHILE handover
            
            if t_current > t_max_w_antenna:
                raise RuntimeError("Failed to find valid handover position within maximum time limit")
            
            # If we didn't find a valid handover position, use the final position
            if handover_position is None:
                raise RuntimeError("Failed to find valid handover position within maximum time limit")
                #handover_position = p_current.copy()

            # Store results
            handover_points[k_current] = handover_position
            
            # Compute final distances and times
            D_pickup_to_handover = la.norm(handover_position - p_pickup)
            t_handover = t_current
            t_move_dur = t_handover - t_pickup
            t_dur = t_pickup_dur + t_move_dur

            # Handle subsequent agent start times due to time cutting
            time_cut = max(0, t_stop_k[k_current] - t_handover - 1)
            for k in best_relaying_agents[idx+1:]:
                time_cut_k = min(time_cut, t_start_k[k])
                t_start_k[k] -= time_cut_k
                t_stop_k[k] -= time_cut_k

            # Adjust end time and starting times for next agent if it has its init point as relay point
            if idx < k_D:
                time_cut_next_k = min(time_cut, t_start_k[k_next])
                t_stop_k[k_next] -= time_cut_next_k

                t_pre_handover = max(0, t_start_k[k_next] - t_handover - 1)
                relay_within_init = (la.norm(p_agents[k_next] - best_relay_points[k_next]) < sigma)
                if t_pre_handover > 0 and relay_within_init:
                    t_start_k[k_next] = t_handover + 1 

                    # Adjut subsequent agent times
                    for k in best_relaying_agents[idx+2:]:
                        t_start_k[k] -= t_pre_handover
                        t_stop_k[k] -= t_pre_handover

    
            if directed_transmission:
                # Adjust steering so that it is performed at the end instead of the beginning (discounting)
                # Find the last phi position and make the beginning steady until that point
                phi_final = antenna_directions[-1]
                
                # Update initial antenna direction until final direction reached 
                phi_current = phi_start
                n_phi_steps = 0
                phis = []
                #while np.isclose(phi_current, phi_final, atol=1e-2):# and n_phi_steps < 9:
                while (phi_current != phi_final) and (n_phi_steps < 9):
                    phi_current = one_antenna_step(phi_current, phi_target, phi_resolution)
                    phis.append(phi_current)
                    n_phi_steps += 1

                if n_phi_steps > 0:
                    n_phi = len(antenna_directions)
                    new_antenna_directions = np.zeros_like(antenna_directions)
                    n_phi_init = n_phi - n_phi_steps
                    new_antenna_directions[:n_phi_init] = [phi_start] * n_phi_init
                    new_antenna_directions[-n_phi_steps:] = phis
                    
                    antenna_directions = new_antenna_directions
            
                # Sanity check, antenna_directions for each time point the agent is active
                if len(antenna_directions) != (t_dur+1):
                    raise RuntimeError("Antenna directions length mismatch with active time steps")

            # If there is a next agent, adjust its relay point 
            if idx < k_D:
                best_relay_points[k_next] = p_next_agent

            # Handle subsequent agent start times due to time difference (save/cut if directed, increase if jammed)
            t_stop_dif = t_stop_k[k_current] - t_handover
            for k in best_relaying_agents[idx:]:
                if t_start_k[k] >= t_stop_dif:  # can't have negative start time
                    t_start_k[k] -= (t_stop_dif)
                t_stop_k[k] -= (t_stop_dif)

                    
            # Update agent timing
            D_k[k_current] = D_pickup + D_pickup_to_handover
            t_stop_k[k_current] = t_handover

            # Account for antenna steering cost
            if directed_transmission:

                phi_cost_temp = 0
                for i in range(t_dur):  # has the antenna moved from one time step to the next?
                    t_temp = t_start + i

                    if antenna_directions[i] != antenna_directions[i+1]:
                        phi_cost_temp += c_phi * (beta**t_temp)

                phi_costs[k_current] = phi_cost_temp

                antenna_directions_all[k_current] = antenna_directions

            # TODO: For fixed agents, steer antenna in correct position based on start and stop times
            
        # END FOR each agent in relay chain (including retreiving agent)

    # END IF scenarios that are NOT isotropic without jammer

    # Compute pickup times for isotropic no jammer scenario
    if not directed_transmission and not jammer_on:
        for idx in range(k_D + 1):  # +1 to include retrieving agent
            # Determine pickup point
            if idx == 0:
                k_current = best_k
                p_pickup = ps_k
                t_start = 0  
            else:
                k_current = best_relaying_agents[idx - 1]
                p_pickup = best_relay_points[k_current]
                t_start = t_start_k[k_current]

            # Compute time and distance for retrieving agent to reach pickup point from TX base
            D_pickup = la.norm(p_pickup - p_agents[k_current])
            t_pickup_dur = int(np.ceil(D_pickup / sigma)) + 1  # +1 for pickup time
            t_pickup = t_start + t_pickup_dur

            t_pickup_all[k_current] = t_pickup


    # Value computation (see reasoning behind cost parameters, especially state s^#)
    # Compute delivery time
    T_D = int(np.max(t_stop_k[t_stop_k >= 0])) if np.any(t_stop_k >= 0) else 0

    # Compute budget
    budget = compute_budget(K, R, sigma, beta)

    V_piD = (beta**T_D)*budget  
    for t in range(T_D+1):
        for k in range(K):
            if t_start_k[k] <= t < t_stop_k[k]:
                V_piD -= c_pos * sigma**2 * (beta**t)
                
    # Account for antenna cost
    if directed_transmission:
        for k in range(K):
            V_piD -= phi_costs[k]  # already discounted and scaled above
                

    # Compute sum of agent distances
    D_all = np.sum(D_k)

    # Compute distance message has traveled through air
    D_message_air = 0.0

    # From transmitting base to retrieving agent's retrieval point (UNCAPPED)
    if not jammer_on:
        ps_k = best_graph.nodes['s_k']['pos']
    D_message_air += la.norm(ps_k - p_tx)

    # From retrieving agent to first relaying agent (or to RX base if no relays)
    if k_D > 0:
        first_relay = best_relaying_agents[0]
        D_message_air += min(Rcom, la.norm(best_relay_points[first_relay] - ps_k))
        
        # Between relaying agents
        for idx in range(k_D - 1):
            first_relay = best_relaying_agents[idx]
            second_relay = best_relaying_agents[idx+1]
            D_message_air += min(Rcom, la.norm(best_relay_points[second_relay] - handover_points[first_relay]))
        
        # From last relay point to receiving base
        last_relay = best_relaying_agents[-1]
        D_message_air += min(Rcom, la.norm(p_recv - handover_points[last_relay]))
    else:
        # No relaying agents: direct TX → RX
        D_message_air += min(Rcom, la.norm(p_recv - ps_k))


    if debug:
        print(f"Passive agents: {passive_agents}")

    result = {
        'retrieving_agent': best_k,
        'path': best_path,
        'active_agents': all_active,
        'relaying_agents': best_relaying_agents,
        'relay_points': best_relay_points,  # does not contain retrieving agent
        'passive_agents': passive_agents,
        'passive_relay_points': passive_relay_points,
        'graph': best_graph,
        'agent_start_times': t_start_k,
        'agent_stop_times': t_stop_k,
        'distance': best_distance,
        'agent_distances': D_k,
        'handover_points': handover_points,
        'antenna_directions': antenna_directions_all,
        'pickup_times': t_pickup_all,
        'agent_sum_distance': D_all,
        'message_air_distance': D_message_air,
        'delivery_time': T_D,
        'value': V_piD,
        'budget': budget,
        'p_jammer_all': p_jammer_all,
        'v_jammer_all': v_jammer_all,
        'ps_k': ps_k
    }

    return result



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


def plot_scenario(sample, directed=False, jammer=False, savepath=None, debug=False):
    """Plot the initial scenario with agent positions, directions, jammer and bases."""
    R = sample['R']
    p_agents = sample['p_agents']
    p_recv = sample['p_recv']
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

    # Add agent index annotations (same as in plot_scenario_with_path_colored)
    if debug:  # annotate the agent indices
        for i, pos in enumerate(p_agents):
            ax.text(pos[0]+0.2*Rcom, pos[1]+0.2*Rcom, str(i), color='black', fontsize=12, ha='center', va='center')

    # Plot initial agent orientations
    if directed:
        phi_agents = sample['phi_agents']
        for i in range(p_agents.shape[0]):
            phi = phi_agents[i]
            p = p_agents[i]
            #ax.annotate("", xy=p + arrow_scale*sigma*np.array([np.cos(phi), np.sin(phi)]), xytext=p,    # plot displacement vector (jammer motion direction)
            #        arrowprops=dict(arrowstyle="->", color=init_agent_color, lw=line_width))

            # New antenna plot (range pattern)
            # Create theta range for directed transmission
            theta_eps = 0.05
            theta_range_dir = np.linspace(-np.pi/2 + theta_eps, np.pi/2 - theta_eps, 500)
            
            # Calculate communication range for this phi (directed, no jammer)
            SINR_threshold = 1.0  # minimum SINR for communication
            ranges_dir = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, 
                                            jammer_pos=None, p_tx=p) for theta in theta_range_dir]
            
            # Convert theta to actual angles (relative to antenna orientation phi)
            actual_angles_dir = theta_range_dir + phi
            
            # Convert to Cartesian coordinates
            range_x_dir = p[0] + np.array(ranges_dir) * np.cos(actual_angles_dir)
            range_y_dir = p[1] + np.array(ranges_dir) * np.sin(actual_angles_dir)
            
            # Plot the communication range pattern
            fill_alpha = 0.2
            line_alpha = 0.7
            linewidth = 2
            color = "orange"
            plt.fill(range_x_dir, range_y_dir, alpha=fill_alpha, color=color, zorder=i+1)
            plt.plot(range_x_dir, range_y_dir, color=color, linewidth=linewidth, 
                    alpha=line_alpha, zorder=i+1)
            
            # Draw antenna boresight arrows for each direction
            max_range = max(ranges_dir) if ranges_dir else 1.0
            antenna_length = 0.8 * max_range
            antenna_end = p + antenna_length * np.array([np.cos(phi), np.sin(phi)])
            
            plt.arrow(p[0], p[1], antenna_end[0] - p[0], antenna_end[1] - p[1], 
                    head_width=0.04, head_length=0.04, fc=color, ec=color, 
                    linewidth=linewidth, alpha=line_alpha, zorder=i+10)
    else:
        # For isotropic transmission, plot circles around each agent
         # Add Rcom circles around each agent
        for agent_pos in p_agents:
            ax.add_artist(plt.Circle(agent_pos, Rcom, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=0.4, zorder=1))

   
    # Plot jammer
    if jammer:
        jammer_info = sample['jammer_info']
        p_j = jammer_info['p_jammer']
        dp_j = jammer_info['v_jammer']
        ax.scatter(p_j[0], p_j[1], c='red', marker='x', s=marker_size+50, label='Jammer')
        ax.annotate("", xy=p_j + arrow_scale*dp_j, xytext=p_j,    # plot displacement vector (jammer motion direction)
                    arrowprops=dict(arrowstyle="->", color='red', lw=line_width))

        # Plot jammer capsule area
        cap_height = 3 * Rcom
        cap_radius = 1.5 * Rcom

        # Top line of rectangle
        ax.plot([0, R], [cap_height/2, cap_height/2], color='red', lw=2, alpha=0.4)
        
        # Bottom line of rectangle
        ax.plot([0, R], [-cap_height/2, -cap_height/2], color='red', lw=2, alpha=0.4)

        # Left half circle (outer edge only)
        theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
        x_left = cap_radius * np.cos(theta_left)
        y_left = cap_radius * np.sin(theta_left)
        ax.plot(x_left, y_left, color='red', lw=2, alpha=0.4, label='Jammer capsule')

        # Right half circle (outer edge only)
        theta_right = np.linspace(-np.pi/2, np.pi/2, 100)
        x_right = R + cap_radius * np.cos(theta_right)
        y_right = cap_radius * np.sin(theta_right)
        ax.plot(x_right, y_right, color='red', lw=2, alpha=0.4)

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

def plot_scenario_with_path_colored(sample, dijkstra_result, savepath=None, directed_transmission=False, jammer=False, debug=False):
    R = sample['R']
    p_agents = sample['p_agents']
    K = len(p_agents)
    p_recv = sample['p_recv']
    p_tx = sample['p_tx']
    Rcom = sample['Rcom']
    Ra = sample['Ra']

    # Colors and line widths
    agent_color = "orange"
    init_agent_color = "grey"
    receive_color = "blue"
    deliver_color = "green"
    passive_color = "black"
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
    norm_d = la.norm(direction)
    end_point = p_recv
    ax.plot([ps_k[0], end_point[0]], [ps_k[1], end_point[1]],
            c=init_agent_color, linestyle='--', linewidth=line_width, alpha=0.6, zorder=1)

    # Instead of plotting the retrieving point, plot the point where the retrieving agent relays to the next agent:
    if dijkstra_result['relaying_agents']:
        first_relay = dijkstra_result['relaying_agents'][0]
        start = ps_k
        end = dijkstra_result['relay_points'][first_relay]
        direction = end - start
        norm_d = la.norm(direction)
        if norm_d > 1e-8:
            direction = direction / norm_d
            relay_contact = end - direction * sample['Rcom']
            ax.scatter(relay_contact[0], relay_contact[1], c=agent_color, marker='o', s=marker_size)

    path = dijkstra_result['path']
    relay_points = dijkstra_result['relay_points']
    handovers = dijkstra_result['handover_points']

    # Track if relay points and passive agents are plotted for legend
    relay_point_plotted = False
    passive_agent_plotted = False

    # Plot relay points (destinations) with full opacity, only for relaying agents
    for idx in dijkstra_result['relaying_agents']:
        #relay = relay_points[idx]
        handover = handovers[idx]
        ax.scatter(handover[0], handover[1], c=agent_color, alpha=1.0, marker='o', s=marker_size, zorder=20)
        # Add Rcom circle around relay point
        circle = plt.Circle(handover, Rcom, color=agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
        ax.add_artist(circle)
        relay_point_plotted = True
    
    if debug:
        passive_agents = dijkstra_result['passive_agents']
        passive_relay_points = dijkstra_result['passive_relay_points']
        for idx in passive_agents:
            relay = passive_relay_points[idx]
            ax.scatter(relay[0], relay[1], c=init_agent_color, alpha=1.0, marker='o', s=marker_size, zorder=20)
            circle = plt.Circle(relay, Rcom, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
            ax.add_artist(circle)
            ax.annotate(
                "",  # No text
                xy=relay,
                xytext=p_agents[idx],
                arrowprops=dict(
                    arrowstyle="->",
                    color=init_agent_color,
                    lw=line_width,
                    alpha=alpha_init_agent
                )
            )
            relay_point_plotted = True

    # Plot initial agent positions
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size, zorder=21)
    
    # Plot passive agents in red at their initial positions
    for idx in range(len(p_agents)):
        if idx not in dijkstra_result['relaying_agents'] and idx != dijkstra_result['retrieving_agent']:
            ax.scatter(p_agents[idx, 0], p_agents[idx, 1], c=passive_color, alpha=1.0, marker='o', s=marker_size, zorder=22)
            passive_agent_plotted = True

    if debug:  # annotate the agent indices
        for i, pos in enumerate(p_agents):
            ax.text(pos[0]+0.2*Rcom, pos[1]+0.2*Rcom, str(i), color='black', fontsize=12, ha='center', va='center')

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
            if isinstance(node, int) and node in handovers:
                #return relay_points[node]
                return handovers[node]
            else:
                return node_positions[node]
        start = get_plot_pos(node_from)
        end = get_plot_pos(node_to)
        # If next node is a relay agent, plot arrow to the edge of its Rcom circle
        if isinstance(node_to, int) and node_to in handovers:
            direction = handovers[node_to] - start
            norm_d = la.norm(direction)
            # Only plot arrow if distance > Rcom
            if norm_d > Rcom + 1e-8:
                direction = direction / norm_d
                end = handovers[node_to] - direction * sample['Rcom']
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))
        elif node_to == 'r':
            # Last arrow: from last relay point to edge of receiver base Rcom circle
            direction = p_recv - start
            norm_d = la.norm(direction)
            if norm_d > Rcom + 1e-8:
                direction = direction / norm_d
                #end = p_recv - direction * sample['Rcom']
                # Get the last integer in the path (one-liner)
                last_agent = next((node for node in reversed(path) if isinstance(node, int)), None)
                end = handovers[last_agent]  # handover point to receiving base
                last_meet_point = end  # Save the final relay point
                ax.annotate("", xy=end, xytext=start,
                            arrowprops=dict(arrowstyle="->", color=deliver_color, lw=line_width))
        else:
            # For other cases, plot arrow if distance > Rcom
            norm_d = la.norm(end - start)
            if norm_d > Rcom + 1e-8:
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

    # Plot final antenna configurations
    if directed_transmission:
        # For every relay agent and the retrieving agent, plot the final antenna direction at their handover points
        antenna_directions = dijkstra_result['antenna_directions']
        for k in antenna_directions.keys():
            phis_k = antenna_directions[k]
            phi_last = phis_k[-1]
            handover_k = handovers[k]

            # Create theta range for directed transmission
            theta_eps = 0.05
            theta_range_dir = np.linspace(-np.pi/2 + theta_eps, np.pi/2 - theta_eps, 500)
            
            # Calculate communication range for this phi (directed, no jammer)
            SINR_threshold = 1.0  # minimum SINR for communication
            ranges_dir = [communication_range(theta, phi_last, C_dir=1.0, SINR_threshold=SINR_threshold, 
                                            jammer_pos=None, p_tx=handover_k) for theta in theta_range_dir]
            
            # Convert theta to actual angles (relative to antenna orientation phi)
            actual_angles_dir = theta_range_dir + phi_last
            
            # Convert to Cartesian coordinates
            range_x_dir = handover_k[0] + np.array(ranges_dir) * np.cos(actual_angles_dir)
            range_y_dir = handover_k[1] + np.array(ranges_dir) * np.sin(actual_angles_dir)
            
            # Plot the communication range pattern
            fill_alpha = 0.2
            line_alpha = 0.7
            linewidth = 2
            color = "orange"
            plt.fill(range_x_dir, range_y_dir, alpha=fill_alpha, color=color, zorder=i+1)
            plt.plot(range_x_dir, range_y_dir, color=color, linewidth=linewidth, 
                    alpha=line_alpha, zorder=i+1)
            
            # Draw antenna boresight arrows for each direction
            max_range = max(ranges_dir) if ranges_dir else 1.0
            antenna_length = 0.8 * max_range
            antenna_end = handover_k + antenna_length * np.array([np.cos(phi_last), np.sin(phi_last)])
            
            plt.arrow(handover_k[0], handover_k[1], antenna_end[0] - handover_k[0], antenna_end[1] - handover_k[1], 
                    head_width=0.04, head_length=0.04, fc=color, ec=color, 
                    linewidth=linewidth, alpha=line_alpha, zorder=i+10)

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
    else:
        plt.show()


# Replace the animate function within animate_scenario_with_path_colored with:

def animate_scenario_with_path_colored(sample, dijkstra_result, directed_transmission=False, jammer_on=False, savepath=None, beamer_gif=False, interval=100):
    
    if savepath and beamer_gif:
        png_folder = savepath[:-4] + "_frames" 
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)

    p_agents = sample['p_agents']
    sigma = sample['sigma']
    K = p_agents.shape[0]
    t_start_k = dijkstra_result['agent_start_times']
    t_stop_k = dijkstra_result['agent_stop_times']
    T_D = dijkstra_result['delivery_time']
    relay_points = dijkstra_result['relay_points']
    relaying_agents = dijkstra_result['relaying_agents']
    retrieving_agent = dijkstra_result['retrieving_agent']
    handover_points = dijkstra_result['handover_points']
    pickup_times = dijkstra_result['pickup_times']

    if directed_transmission:
        C_dir = 1.0
    else:
        C_dir = 0.0

    if jammer_on:
        p_jammer_all = dijkstra_result['p_jammer_all']
        v_jammer_all = dijkstra_result['v_jammer_all']
    
    # Get antenna directions if available
    antenna_directions = dijkstra_result.get('antenna_directions', {}) if directed_transmission else {}

    # Colors and line widths
    agent_color = "orange"
    init_agent_color = "grey"
    send_color = "blue"
    pickup_color = "green"
    passive_color = "black"
    alpha_circle = 0.6
    alpha_init_agent = 0.6
    line_width = 2
    marker_size = 80

    R = sample['R']
    Rcom = sample['Rcom']
    Ra = sample['Ra']
    p_tx = sample['p_tx']
    p_recv = sample['p_recv']

    fig, ax = plt.subplots(figsize=(8, 8))  # Increased size for better visibility
    
    # ... [Keep all the existing static plotting code unchanged] ...
    
    # Plot initial agent positions in grey (static, underneath everything)
    ax.scatter(p_agents[:, 0], p_agents[:, 1], c=init_agent_color, alpha=alpha_init_agent, s=marker_size, zorder=1, label='Init pos')

    # Static elements 
    ax.scatter(*p_tx, c=send_color, marker='s', s=marker_size, label='TX base')
    ax.scatter(*p_recv, c=pickup_color, marker='s', s=marker_size, label='RX base')
    if not jammer_on:
        ax.add_artist(plt.Circle(p_recv, Rcom, color=pickup_color, fill=False, linestyle='--', lw=line_width, alpha=alpha_circle))
    ax.add_artist(plt.Circle([R/2, 0], 0.6*R, color=init_agent_color, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle))

    # Plot jammer capsule area
    if jammer_on:
        cap_height = 3 * Rcom
        cap_radius = 1.5 * Rcom

        # Top line of rectangle
        ax.plot([0, R], [cap_height/2, cap_height/2], color='red', lw=2, alpha=0.4)
        
        # Bottom line of rectangle
        ax.plot([0, R], [-cap_height/2, -cap_height/2], color='red', lw=2, alpha=0.4)

        # Left half circle (outer edge only)
        theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
        x_left = cap_radius * np.cos(theta_left)
        y_left = cap_radius * np.sin(theta_left)
        ax.plot(x_left, y_left, color='red', lw=2, alpha=0.4, label='Jammer capsule')

        # Right half circle (outer edge only)
        theta_right = np.linspace(-np.pi/2, np.pi/2, 100)
        x_right = R + cap_radius * np.cos(theta_right)
        y_right = cap_radius * np.sin(theta_right)
        ax.plot(x_right, y_right, color='red', lw=2, alpha=0.4)

    # Add retrieval point for the retrieving agent
    km = retrieving_agent

    # Plot dashed line from retrieval point to the Rcom circle of the receiving base
    ps_k = dijkstra_result['ps_k']
    if not jammer_on:
        ps_k = compute_retrieval_point(p_agents[km], sample['Rcom'])
        direction = p_recv - ps_k
        norm_d = la.norm(direction)
        if norm_d > 1e-8:
            direction = direction / norm_d
            end_point = p_recv - direction * Rcom
        else:
            end_point = p_recv
        ax.plot([ps_k[0], end_point[0]], [ps_k[1], end_point[1]],
                c=init_agent_color, linestyle='--', linewidth=line_width, alpha=0.6, zorder=1)

    # Build waypoints for each agent 
    waypoints = [None] * K
    for k in range(K):
        if k == retrieving_agent:
            # Retrieving agent: initial -> retrieval point -> first relay point or receiver Rcom
            if la.norm(ps_k - handover_points[k]) < sigma:
                w = [p_agents[k], ps_k, handover_points[k]]
            else:
                w = [p_agents[k], ps_k, handover_points[k]]
        elif k in relay_points:
            if la.norm(relay_points[k] - handover_points[k]) < sigma:
                w = [p_agents[k], relay_points[k]]
            else:
                w = [p_agents[k], relay_points[k], handover_points[k]]
        else:  # passive agent
            w = [p_agents[k]]

        waypoints[k] = w

    # Agent position interpolation 
    def get_agent_pos(k, t):
        t_start = t_start_k[k]
        t_stop = t_stop_k[k]
        wp = waypoints[k]
        n_wp = len(wp)
        if n_wp == 1:
            return wp[0]
        elif n_wp == 2:
            seg1 = la.norm(wp[1] - wp[0])
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
            seg1 = la.norm(wp[1] - wp[0])
            seg2 = la.norm(wp[2] - wp[1])
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

    # Function to get antenna direction at time t for agent k
    def get_antenna_direction(k, t):
        if k not in antenna_directions:
            return sample.get('phi_agents', np.zeros(K))[k]  # Default to initial direction
        
        phis_k = antenna_directions[k]
        t_start = t_start_k[k]
        t_stop = t_stop_k[k]
        
        if t < t_start:
            # Before agent starts moving - use initial direction
            return sample.get('phi_agents', np.zeros(K))[k]
        elif t >= t_stop:
            # After agent stops - use final direction
            return phis_k[-1] if len(phis_k) > 0 else sample.get('phi_agents', np.zeros(K))[k]
        else:
            # During movement - interpolate through antenna directions
            time_index = min(int(t - t_start), len(phis_k) - 1)
            return phis_k[time_index] if time_index >= 0 and time_index < len(phis_k) else sample.get('phi_agents', np.zeros(K))[k]

    # Animation objects 
    agent_dots = ax.scatter(p_agents[:, 0], p_agents[:, 1], c=[agent_color]*K, s=marker_size, zorder=30)
    relay_arrows1 = []
    relay_arrows2 = []
    for _ in relaying_agents:
        arrow1 = ax.annotate("", xy=(0,0), xytext=(0,0),
                             arrowprops=dict(arrowstyle="->", color=init_agent_color, lw=line_width, alpha=alpha_init_agent))
        relay_arrows1.append(arrow1)
        arrow2 = ax.annotate("", xy=(0,0), xytext=(0,0),
                             arrowprops=dict(arrowstyle="->", color=pickup_color, lw=line_width, alpha=1.0))
        relay_arrows2.append(arrow2)

        
    retrieving_arrow1 = ax.annotate("", xy=(0,0), xytext=(0,0),
                                    arrowprops=dict(arrowstyle="->", color=pickup_color, lw=line_width, alpha=1.0))
    retrieving_arrow2 = ax.annotate("", xy=(0,0), xytext=(0,0),
                                    arrowprops=dict(arrowstyle="->", color=pickup_color, lw=line_width, alpha=1.0))
    
    # TX Rcom circle varies depending on jammer
    if not jammer_on:
        ax.add_artist(plt.Circle(p_tx, Rcom, color=send_color, fill=False, linestyle='--', lw=line_width, alpha=alpha_circle))

    TX_Rcom_circle = None
    
    # Initialize jammer scatter plot
    jammer_scatter = None
    jammer_arrow = None

    # Initialize persistent circles per agent
    agent_circles_persistent = {}  # Store persistent circles for each agent
    

    # Animation update function 
    def animate(t):
        nonlocal jammer_scatter, jammer_arrow, TX_Rcom_circle

        # Remove previous Rcom circles and range patterns
        for c in getattr(ax, "_agent_circles", []):
            try:
                c.remove()
            except Exception:
                pass

        ax._agent_circles = []

        # Re-add persistent circles that should still be visible
        for k, circ in agent_circles_persistent.items():
            if t >= t_stop_k[k]:
                ax._agent_circles.append(circ)
        
        
        # Remove previous range patterns
        for artist in getattr(ax, "_range_patterns", []):
            try:
                artist.remove()
            except Exception:
                pass
        ax._range_patterns = []

        # Remove previous dynamic relay point markers
        for artist in getattr(ax, "_dynamic_relay_points", []):
            try:
                artist.remove()
            except Exception:
                pass
        ax._dynamic_relay_points = []

        # Remove previous jammer visualization
        if jammer_scatter is not None:
            try:
                jammer_scatter.remove()
            except Exception:
                pass
        if jammer_arrow is not None:
            try:
                jammer_arrow.remove()
            except Exception:
                pass
        if TX_Rcom_circle is not None:
            try:
                if t <= pickup_times[retrieving_agent]:
                    TX_Rcom_circle.remove()
            except Exception:
                pass

        current_pos = np.zeros_like(p_agents)
        colors = []
        
        # Plot agents and range patterns
        for k in range(K):
            pos = get_agent_pos(k, t)
            current_pos[k] = pos  # Store the current position

            
            if k == retrieving_agent or k in relaying_agents:
                
                # Coloring the agents depending on received message received
                if k in pickup_times.keys():
                    if t >= pickup_times[k]:
                        colors.append(pickup_color)
                    else:
                        colors.append(agent_color)
                else:
                        colors.append(passive_color)

                # 
                if t >= pickup_times[k]:
                    color_com = pickup_color
                else:
                    color_com = agent_color
                
                # Draw Rcom circle only while moving AND only if NOT using directed transmission
                if not directed_transmission:
                    if t <= t_stop_k[k]:  # Only draw circle while agent is moving
                        if jammer_on:
                            RX_Rcom = communication_range(0, 0, C_dir=0.0, SINR_threshold=1.0, jammer_pos=p_jammer_all[t], p_tx=pos)
                            circ = plt.Circle(pos, RX_Rcom, color=color_com, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)
                        else:
                            circ = plt.Circle(pos, Rcom, color=color_com, fill=False, linestyle='--', linewidth=line_width, alpha=alpha_circle, zorder=2)

                        ax.add_artist(circ)
                        ax._agent_circles.append(circ)
                        agent_circles_persistent[k] = circ  # Store for persistence
                    elif k in agent_circles_persistent and t >= t_stop_k[k]:
                        # Keep the persistent circle - it's already on the plot
                        ax._agent_circles.append(agent_circles_persistent[k])
                
                # Add directed transmission range patterns
                if directed_transmission and k in antenna_directions:
                    phi_current = get_antenna_direction(k, t)
                    
                    # Create theta range for directed transmission
                    theta_eps = 0.05
                    theta_range_dir = np.linspace(-np.pi/2 + theta_eps, np.pi/2 - theta_eps, 100)
                    
                    # Calculate communication range for this phi (directed, no jammer)
                    SINR_threshold = 1.0  # minimum SINR for communication
                    try:
                        ranges_dir = [communication_range(theta, phi_current, C_dir=C_dir, SINR_threshold=SINR_threshold, 
                                                        jammer_pos=None, p_tx=pos) for theta in theta_range_dir]
                        
                        # Convert theta to actual angles (relative to antenna orientation phi)
                        actual_angles_dir = theta_range_dir + phi_current
                        
                        # Convert to Cartesian coordinates
                        range_x_dir = pos[0] + np.array(ranges_dir) * np.cos(actual_angles_dir)
                        range_y_dir = pos[1] + np.array(ranges_dir) * np.sin(actual_angles_dir)
                        
                        # Plot the communication range pattern - ALL ORANGE AND FILLED
                        fill_alpha = 0.2
                        line_alpha = 0.7
                        range_linewidth = 1.5
                        range_color = color_com  # Same orange color for all agents
                        
                        # Add filled pattern
                        fill_patch = ax.fill(range_x_dir, range_y_dir, alpha=fill_alpha, color=range_color, zorder=3)[0]
                        ax._range_patterns.append(fill_patch)
                        
                        # Add outline
                        line_patch, = ax.plot(range_x_dir, range_y_dir, color=range_color, linewidth=range_linewidth, 
                                             alpha=line_alpha, zorder=4)
                        ax._range_patterns.append(line_patch)
                        
                        # Draw antenna boresight arrow
                        max_range = max(ranges_dir) if ranges_dir else 1.0
                        antenna_length = 0.6 * max_range
                        antenna_end = pos + antenna_length * np.array([np.cos(phi_current), np.sin(phi_current)])
                        
                        arrow_patch = ax.annotate("", xy=antenna_end, xytext=pos,
                                                 arrowprops=dict(arrowstyle="->", color=range_color, 
                                                               lw=range_linewidth+0.5, alpha=line_alpha))
                        ax._range_patterns.append(arrow_patch)
                        
                    except Exception as e:
                        # If communication_range fails, skip this pattern
                        pass
            else:
                colors.append(passive_color)

        # END FOR agents 

        agent_dots.set_offsets(current_pos)  # shift agent positions
        agent_dots.set_facecolors(colors)

        # Plot jammer if available
        if jammer_on:
            p_jammer_t = p_jammer_all[t]
            v_jammer_t = v_jammer_all[t]
            
            # Plot jammer as red dot
            jammer_scatter = ax.scatter(p_jammer_t[0], p_jammer_t[1], c='red', marker='o', s=marker_size+50, zorder=35)
            
            # Plot jammer velocity arrow
            arrow_scale = 3
            jammer_arrow = ax.annotate("", xy=p_jammer_t + arrow_scale*v_jammer_t, xytext=p_jammer_t,
                       arrowprops=dict(arrowstyle="->", color='red', lw=line_width))

            # Plot jammer path (trajectory up to current time)
            if t > 0:
                jammer_path_x = [p_jammer_all[i][0] for i in range(t+1)]
                jammer_path_y = [p_jammer_all[i][1] for i in range(t+1)]
                ax.plot(jammer_path_x, jammer_path_y, color='red', linestyle='-', linewidth=1, alpha=0.5, zorder=34)


            # Update TX Rcom circle with jammer effect
            if t <= pickup_times[retrieving_agent]:
                TX_Rcom = communication_range(0, 0, C_dir=C_dir, SINR_threshold=1.0, jammer_pos=p_jammer_t)
                TX_Rcom_circle = ax.add_artist(plt.Circle(p_tx, TX_Rcom, color=send_color, fill=False, linestyle='--', lw=line_width, alpha=alpha_circle))


        # Relaying agent arrows (two segments) 
        for idx, k in enumerate(relaying_agents):
            wp = waypoints[k]
            pos = get_agent_pos(k, t)
            if len(wp) == 3:
                seg1 = la.norm(wp[1] - wp[0])
                seg2 = la.norm(wp[2] - wp[1])
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
                    relay_arrows2[idx].arrow_patch.set_color(pickup_color)
                    relay_arrows2[idx].arrow_patch.set_alpha(1)
            elif len(wp) == 2:
                relay_arrows1[idx].set_position(wp[0])
                relay_arrows1[idx].xy = pos
                relay_arrows1[idx].arrow_patch.set_color(init_agent_color)
                relay_arrows1[idx].arrow_patch.set_alpha(alpha_init_agent)
                relay_arrows2[idx].set_position((0,0))
                relay_arrows2[idx].xy = (0,0)
                relay_arrows2[idx].arrow_patch.set_alpha(0)

        # Retrieving agent arrows (two segments) 
        wp = waypoints[retrieving_agent]
        pos = get_agent_pos(retrieving_agent, t)
        if len(wp) == 3:
            seg1 = la.norm(wp[1] - wp[0])
            seg2 = la.norm(wp[2] - wp[1])
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
                retrieving_arrow1.arrow_patch.set_color(pickup_color)
                retrieving_arrow1.arrow_patch.set_alpha(1)
                retrieving_arrow2.set_position((0,0))
                retrieving_arrow2.xy = (0,0)
                retrieving_arrow2.arrow_patch.set_alpha(0)
            else:
                retrieving_arrow1.set_position(wp[0])
                retrieving_arrow1.xy = wp[1]
                retrieving_arrow1.arrow_patch.set_color(pickup_color)
                retrieving_arrow1.arrow_patch.set_alpha(1)
                retrieving_arrow2.set_position(wp[1])
                retrieving_arrow2.xy = pos
                retrieving_arrow2.arrow_patch.set_color(pickup_color)
                retrieving_arrow2.arrow_patch.set_alpha(1)
        elif len(wp) == 2:
            retrieving_arrow1.set_position(wp[0])
            retrieving_arrow1.xy = pos
            retrieving_arrow1.arrow_patch.set_color(pickup_color)
            retrieving_arrow1.arrow_patch.set_alpha(1)
            retrieving_arrow2.set_position((0,0))
            retrieving_arrow2.xy = (0,0)
            retrieving_arrow2.arrow_patch.set_alpha(0)

        ax.set_title(f"Dijkstra Baseline\n ($K$: {K}; Time: {t} / {max(int(np.max(t_stop_k)+1), T_D)} Value: {dijkstra_result['value']:.2f})")
        # OBS! In np.max(t_stop_k)+1 and T_D should be equal!


        # Save PNG snapshot if requested
        if savepath is not None and beamer_gif:
            fname = os.path.join(png_folder, f"frame_{t:04d}.png")
            plt.savefig(fname, dpi=150)

        return [agent_dots] + relay_arrows1 + relay_arrows2 + [retrieving_arrow1, retrieving_arrow2]

    # Legend 
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='TX base', markerfacecolor=send_color, markersize=10),
        Line2D([0], [0], marker='s', color='w', label='RX base', markerfacecolor=pickup_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='No message', markerfacecolor=agent_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Has message', markerfacecolor=pickup_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Passive agent', markerfacecolor=passive_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Jammer', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color=pickup_color, lw=2, label='Relay path'),
    ]
    
    if directed_transmission:
        legend_elements.append(Line2D([0], [0], color="orange", lw=2, label='Comm range', alpha=0.6))
    
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_aspect('equal')
    x_lim_max = max(0.4*Ra, 1.6*Rcom)
    ax.set_xlim(-x_lim_max, R + x_lim_max)
    ax.set_ylim(-1.3 * Ra, 1.3 * Ra)
    

    if jammer_on:
        ani = animation.FuncAnimation(
            fig, animate, frames=int(np.max(t_stop_k)) + 1, interval=interval, blit=False, repeat=False
        )
    else:
        ani = animation.FuncAnimation(
            fig, animate, frames=int(np.max(t_stop_k)) + 2, interval=interval, blit=False, repeat=False
        )

    if savepath:
        ani.save(savepath, writer='pillow', fps=1000//interval)
        print(f"Saved animation in {savepath}")
    else:
        plt.show()


def load_premade_scenario(option=1, Rcom=1.0, sigma=0.1):
    """
    Returns a dictionary in the same format as sample_scenario, but with premade agent positions.
    option: int, selects which premade scenario to load.
    """
    if option == 1:
        K = 3
        x_shift = 5.9*Rcom
        R = (K + 4) * Rcom
        Ra = 0.6 * R
        p_tx = np.array([0.0, 0.0])
        p_recv = np.array([R, 0.0])
        p0 = np.array([R/2, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [x_shift*Rcom, Rcom],
            [(x_shift+0.5)*Rcom, 1.5*Rcom]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 2:
        K = 4
        x_shift = 4.8*Rcom
        y = 2*Rcom
        a_shift = 0.8*Rcom
        R = (K + 4) * Rcom
        Ra = 0.6 * R
        p_tx = np.array([0.0, 0.0])
        p_recv = np.array([R, 0.0])
        p0 = np.array([R/2, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [x_shift*Rcom, y],
            [(x_shift+a_shift)*Rcom, y+0.5*Rcom],
            [(x_shift+2*a_shift)*Rcom, y]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 3:
        K = 4
        x_shift = 6*Rcom
        y = 2*Rcom
        a_shift = 0.8*Rcom
        R = (K + 4) * Rcom
        Ra = 0.6 * R
        p_tx = np.array([0.0, 0.0])
        p_recv = np.array([R, 0.0])
        p0 = np.array([R/2, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [x_shift*Rcom, y],
            [(x_shift+a_shift)*Rcom, y+0.5*Rcom],
            [(x_shift+2*a_shift)*Rcom, y]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 4:
        K = 7
        x_shift1 = 5.5*Rcom
        a_shift = 0.6*Rcom
        x_shift2 = x_shift1+2.5*Rcom
        R = (K + 4) * Rcom
        Ra = 0.6 * R
        p_tx = np.array([0.0, 0.0])
        p_recv = np.array([R, 0.0])
        p0 = np.array([R/2, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [x_shift1*Rcom, Rcom],
            [(x_shift1+0.5)*Rcom+a_shift, -1.5*Rcom],
            [(x_shift1+1)*Rcom+2*a_shift, Rcom],
            [x_shift2*Rcom, -Rcom],
            [(x_shift2+0.5)*Rcom+a_shift, 1.5*Rcom],
            [(x_shift2+1)*Rcom+2*a_shift, -Rcom]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 5:
        K = 8
        x_shift1 = 5.5*Rcom
        a_shift = 0.6*Rcom
        x_shift2 = x_shift1+2.5*Rcom
        R = (K + 4) * Rcom
        Ra = 0.6 * R
        p_tx = np.array([0.0, 0.0])
        p_recv = np.array([R, 0.0])
        p0 = np.array([R/2, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [x_shift1*Rcom, Rcom],
            [(x_shift1+0.5)*Rcom+a_shift, -1.5*Rcom],
            [(x_shift1+1)*Rcom+2*a_shift, Rcom],
            [x_shift2*Rcom, -Rcom],
            [(x_shift2+0.5)*Rcom+a_shift, 1.5*Rcom],
            [(x_shift2+0.5)*Rcom+a_shift+0.2, -2*Rcom],
            [(x_shift2+1)*Rcom+2*a_shift, -Rcom]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 6:  # s# --- difficult scenario
        K = 3
        R = (K + 4) * Rcom
        x = 1.1*R
        p_tx = np.array([0.0, 0.0])
        Ra = 0.6 * R
        p0 = np.array([R/2, 0.0])
        p_recv = np.array([R, 0.0])
        p_agents = np.array([
            [x, 0.0],
            [x, 0.0],
            [x, 0.0]
        ])
        phi_agents = np.zeros(K)  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)

    elif option == 7:  # test directed transmission timings with K=2
        K = 2
        R = (K + 1) * Rcom
        x = 1.1*R
        p_tx = np.array([0.0, 0.0])
        Ra = 0.6 * R
        p0 = np.array([R/2, 0.0])
        p_recv = np.array([R, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [2*Rcom, 0.0]
        ])
        phi_agents = np.array([0.0, 0.0])  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)
    elif option == 8:  # test directed transmission timings with K=2
        K = 2
        R = (K + 1) * Rcom
        x = 1.1*R
        p_tx = np.array([0.0, 0.0])
        Ra = 0.6 * R
        p0 = np.array([R/2, 0.0])
        p_recv = np.array([R, 0.0])
        p_agents = np.array([
            [Rcom, 0.0],
            [2*Rcom, 0.0]
        ])
        phi_agents = np.array([0.0, np.pi])  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)
    else:
        raise ValueError("Unknown scenario option")

    return {
            'R': R,
            'Rmin': K * Rcom,
            'Rmax': (K + 4) * Rcom,
            'dense': R <= (K + 1) * Rcom,
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


if __name__ == '__main__':
    debug = True
    test_scenario = 8

    # Baseline
    clustering = True
    eps = 1e-3

    # Environment parameters
    Rcom = 1.0
    sigma = 0.1
    beta = 0.99

    # Cost parameters
    cost_motion = 0.5
    cost_antenna = 0.1

    # Define scenario
    directed = True
    jammer = None

    seed=1

    # Saving stuff
    plot_folder = "Plots"
    init_fig_folder = "Initial_states"
    dijk_fig_folder = "Dijkstra_plots\\7nov"
    anim_folder = "Animations\\7nov"
    beamer_gif = False  # saves png for each frame -> make Beamer gif

    # Scenario parameters
    if test_scenario > 0:
        sample = load_premade_scenario(option=test_scenario, Rcom=1.0, sigma=0.1)
        sample['sigma'] = sigma
        
        file_name = fr"test\\test_{test_scenario}_dijkstra_cluster_baseline_seed{seed}"
        animation_save_path = os.path.join(anim_folder, file_name+'.gif')
        
        result = dijkstra_baseline(sample['p_agents'], 
                                   sample['p_tx'], 
                                   sample['p_recv'], 
                                   Rcom=Rcom, 
                                   sigma=sigma, 
                                   beta=beta, 
                                   phi_agents=sample['phi_agents'], 
                                   c_pos=cost_motion, 
                                   jammer_info=jammer, 
                                   debug=debug, 
                                   clustering=clustering, 
                                   epsilon=eps)    
        
        #plot_scenario(sample, directed=directed, jammer=jammer, savepath=None)

        #plot_scenario_with_path_colored(sample, result, savepath=None, 
        #                                directed_transmission=directed, 
        #                                jammer=jammer, debug=debug)
        animate_scenario_with_path_colored(sample, result, directed_transmission=directed, 
                                           savepath='Animations/test.gif', beamer_gif=True)
    else:
        K_start = 1
        K_stop = 20
        step = 1
        for K in range(K_start,K_stop+1,step):
            R = (K+4)*Rcom
            Ra = 0.6*R

            file_name = fr"dijkstra_baseline_K{K}_seed{seed}"

            #save_path = None
            init_save_path = os.path.join(plot_folder, init_fig_folder, file_name+'_init'+'.png') 
            dijkstra_save_path = os.path.join(plot_folder, dijk_fig_folder, file_name+'.png') 
            animation_save_path = os.path.join(anim_folder, file_name+'.gif')

            sample = sample_scenario(K=K, Rcom=Rcom, R=R, Ra=Ra, sigma=sigma, seed=seed)

            # Manually remove an agents to create a hole in the network
            #i_remove = 6
            #sample['p_agents'] = np.delete(sample['p_agents'], i_remove, axis=0)
            #if 'phi_agents' in sample:
            #    sample['phi_agents'] = np.delete(sample['phi_agents'], i_remove, axis=0)

            plot_scenario(sample, directed=directed, jammer=jammer, savepath=init_save_path)
            #result = dijkstra_baseline(sample['p_agents'], sample['p_tx'], sample['p_recv'], Rcom=Rcom, sigma=sigma, beta=beta, c_pos=cost_motion, jammer=jammer, debug=debug, clustering=clustering, epsilon=eps)    
            #plot_scenario_with_path_colored(sample, result, savepath=None, jammer=jammer, debug=debug)
            #animate_scenario_with_path_colored(sample, result, savepath=animation_save_path, beamer_gif=beamer_gif)