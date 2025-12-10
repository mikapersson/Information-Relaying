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
from Baseline.communication import communication_range, calculate_sinr_at_receiver
from compute_metrics_budget import compute_budget_from_poly, compute_metrics
from Baseline.reflection import optimal_circle_boundary_point


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


def compute_retrieval_point(pk, Rcom, p_tx=np.array([0.0, 0.0]), p_r=None, jammer_on=False):
    """Compute the retrieval point ps_k for agent k given its position pk."""

    if jammer_on:
        v = pk - p_tx 
        norm_pk = la.norm(v)
        if norm_pk > Rcom:
            return (Rcom / norm_pk) * v + p_tx
        else:
            return pk
    else:
        ps_k, _ = optimal_circle_boundary_point(pk, p_r, p_tx, Rcom)  # checks if p_k is inside Rcom of p_tx

        return ps_k


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


#def compute_budget(K, R, sigma, beta):
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


#def compute_metrics(p_trajectories, phi_trajectories, c_pos, c_phi, budget, beta):
    """
    Compute metrics (value, delivery time, and total distance) given agent trajectories.
    
    Args:
        p_trajectories: dict mapping agent index k to dict of positions at each time t
        phi_trajectories: dict mapping agent index k to list/dict of antenna directions
        c_pos: cost coefficient for position movement
        c_phi: cost coefficient for antenna steering
        budget: allocated budget for the scenario
    
    Returns:
        value: scalar value metric (budget - total cost)
        T_D: delivery time (time when message reaches receiver)
        D_tot: total distance traveled by all agents
    """
    
    K = len(p_trajectories)
    
    # Determine total time steps
    T_total = 0
    for k in range(K):
        if k in p_trajectories:
            T_total = max(T_total, max(p_trajectories[k].keys()))
    
    # Compute total distance traveled by all agents
    D_tot = 0.0
    for k in range(K):
        if k in p_trajectories:
            p_traj_k = p_trajectories[k]
            for t in range(len(p_traj_k)-1):
                if t in p_traj_k and (t + 1) in p_traj_k:
                    distance_step = la.norm(p_traj_k[t + 1] - p_traj_k[t])
                    D_tot += distance_step
    
    # Compute antenna steering cost
    phi_cost_total = 0.0
    if c_phi > 0:
        for k in range(K):
            if k in phi_trajectories:
                phi_traj_k = phi_trajectories[k]
                
                # Sum antenna steering costs (only when antenna direction changes)
                times_k = list(p_traj_k.keys())
                for t in times_k[:-1]:
                    phi_current = phi_traj_k[t]
                    phi_next = phi_traj_k[t + 1]
                    phi_diff = np.abs(phi_next - phi_current) if phi_current is not None and phi_next is not None else 0.0
                    
                    # Only count cost if antenna actually moved
                    if phi_current is not None and phi_next is not None:
                        if not np.isclose(phi_current, phi_next, atol=1e-6):
                            phi_cost_total += beta**t * c_phi * phi_diff**2  # Cost per antenna step
    
    # Compute delivery time (T_D)
    # T_D is the time when the message reaches the receiver
    # This is determined by finding the last time step where any agent has the message
    T_D = 0
    for k in range(K):
        if k in p_trajectories:
            p_traj_k = p_trajectories[k]
            T_D = max(T_D, max(p_traj_k.keys()))
    
    T_D = int(T_D) + 1  # Convert to time step (inclusive)
    
    # Compute total movement cost
    # Cost = c_pos * sum of all distances traveled
    position_cost = 0.0
    for k in range(K):
        if k in p_trajectories:
            p_traj_k = p_trajectories[k]
            
            # Sum antenna steering costs (only when antenna direction changes)
            times_k = list(p_traj_k.keys())
            for t in times_k[:-1]:
                p_current = p_traj_k[t]
                p_next = p_traj_k[t + 1]
                p_diff = la.norm(p_next - p_current) if p_current is not None and p_next is not None else 0.0
                
                # Only count cost if antenna actually moved
                if p_current is not None and p_next is not None:
                    position_cost += beta**t * c_pos * p_diff**2  # Cost per antenna step
    
    # Total cost
    total_cost = position_cost + phi_cost_total
    
    # Compute value metric
    # Value = budget - total_cost (higher is better)
    value = budget - total_cost
    
    return value, T_D, D_tot


def baseline(
    p_agents, 
    p_tx, 
    p_recv, 
    Rcom=1.0, 
    sigma=0.2, 
    beta=0.99, 
    c_pos=0.5, 
    c_phi=0.1,
    phi_agents=False, 
    jammer_info=None, 
    debug=False, 
    clustering=False, 
    epsilon=1e-3,
    minimize_distance=False,
    no_metrics=False
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
            ps_k = compute_retrieval_point(pk, Rcom, p_tx, p_recv, jammer_on)
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
        ps_k = compute_retrieval_point(pk, Rcom, p_tx, p_recv, jammer_on)
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
                                if next_proj_dist > Rcom:
                                    next_proj_dist = Rcom
                                t = np.sqrt(Rcom**2 - next_proj_dist**2)
                                #raise RuntimeError("Clustering error: fixed agent too far to maintain contact")
                                
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
                                if prev_proj_dist > Rcom:
                                    prev_proj_dist = Rcom
                                t = np.sqrt(Rcom**2 - prev_proj_dist**2)
                                #raise RuntimeError("Clustering error: fixed agent too far to maintain contact")
                                
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
    D_k[best_k] = la.norm(ps_k - p_agents[best_k]) 
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
    eps = 1e-8
    t_stop_k[best_k] = int(np.ceil((la.norm(ps_k - p_agents[best_k]) - eps) / sigma)) + 1 + int(np.ceil((D_ret_hand - eps) / sigma))
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

        D_to_rel = la.norm(relay_point - p_agents[k]) - eps
        D_to_hand = la.norm(handover_point_k - relay_point) - eps
        D_k[k] = D_to_rel + D_to_hand

        # Start time: must arrive at relay point before the preceding agent arrives
        t_rel_dur = int(np.ceil(D_to_rel / sigma))
        if idx == 0:
            # First relay after retrieving agent
            t_start_k[k] = t_stop_k[best_k] - t_rel_dur# + 1
        else:
            prev_relay = best_relaying_agents[idx-1]
            prev_arrival = t_stop_k[prev_relay]
            t_start_k[k] = prev_arrival - t_rel_dur# + 1

        # Stop time
        t_hand_dur = int(np.ceil(D_to_hand / sigma))
        t_stop_k[k] = t_start_k[k] + t_rel_dur + t_hand_dur + 1
    
            
    # Sanity check, no active agent has reached its relay point before the message could have (excluding air distance for the message)
    if debug:
        if not directed_transmission and not jammer_on:
            for idx, k in enumerate(best_relaying_agents):
                message_distance = la.norm(ps_k - p_agents[best_k]) + \
                    max(0, la.norm(best_relay_points[best_relaying_agents[0]] - ps_k) - Rcom) - eps
                if idx > 0:
                    message_distance += sum(
                        max(0, la.norm(best_relay_points[best_relaying_agents[i+1]] - best_relay_points[best_relaying_agents[i]]) - Rcom) for i in range(idx)  # distance to next relay agent
                    )

                if la.norm(best_relay_points[k] - p_agents[k]) > message_distance:
                    diff = la.norm(best_relay_points[k] - p_agents[k]) - message_distance
                    print(f"Warning: relaying agent {k} could have reached its relay point before the message could have (relay agent moves {diff} further than message)")
        else:
            Warning("Sanity check for agent relay times not implemented for scenarios with orientations or jammers")


    # Save trajectories (positions and antenna directions in all time steps)
    p_trajectories = {}
    phi_trajectories = {}
    has_message = {}
    p_jammer_trajectory = {}

    # Compute trajectories only for isotropic, non-jammed scenario (done for the other scenarios below)
    if not directed_transmission and not jammer_on:
        # Determine total scenario duration
        T_total = int(np.max(t_stop_k)) if np.any(t_stop_k >= 0) else 1

        # Compute trajectories for all agents across all time points
        for k in range(K):
            p_traj_dict = {}
            
            t_intermediate = T_total + 1  # default value for passive agents
            if t_stop_k[k] >= 0:  # active agent
                p_start = p_agents[k]
                
                # Determine intermediate point (relay point or ps_k for retrieving agent)
                if k == best_k:
                    # Retrieving agent: initial -> ps_k -> handover
                    p_intermediate = ps_k
                else:
                    # Relaying agent: initial -> relay_point -> handover
                    p_intermediate = best_relay_points[k]
                
                p_end = handover_points[k]
                
                # Compute distances and times for two segments
                D_to_intermediate = la.norm(p_intermediate - p_start) - eps
                t_to_intermediate_dur = np.ceil(D_to_intermediate / sigma)
                
                D_intermediate_to_end = la.norm(p_end - p_intermediate) - eps
                t_to_end_dur = np.ceil(D_intermediate_to_end / sigma)
                
                D_total = D_to_intermediate + D_intermediate_to_end
                t_total = t_to_intermediate_dur + 1 + t_to_end_dur
                #t_total = t_stop_k[k] - t_start_k[k] - 1
            
                # Time split between segments (proportional to distances)
                if D_total > 1e-12:  # init point same as pickup/intermediate point
                    #t_intermediate = t_start_k[k] + int(np.ceil((D_to_intermediate / D_total) * t_total)) + 1  # time point at which agent reaches intermediate point
                    t_intermediate = t_start_k[k] + t_to_intermediate_dur  # time point at which agent reaches intermediate point
                else:
                    t_intermediate = t_start_k[k]
                
                for t in range(T_total+1):
                    if t <= t_start_k[k]:
                        # Before movement starts: agent at initial position
                        p_traj_dict[t] = p_start.copy()
                    elif t >= t_stop_k[k]:
                        # After movement stops: agent at final position
                        p_traj_dict[t] = p_end.copy()
                    elif t == t_intermediate:
                        # At intermediate point: agent stands still at intermediate position
                        p_traj_dict[t] = p_intermediate.copy()
                    elif t < t_intermediate:
                        # First segment: initial -> intermediate
                        t_elapsed = t - t_start_k[k]
                        if D_to_intermediate == 0:
                            p_traj_dict[t] = p_intermediate.copy()
                        else:
                            #t_seg_total = t_intermediate - t_start_k[k] - 1
                            t_seg_total = t_to_intermediate_dur
                            if t_seg_total > 0:
                                next_dir = (t_elapsed / t_seg_total) * (p_intermediate - p_start)
                                p_traj_dict[t] = p_start + next_dir
                            else:
                                p_traj_dict[t] = p_intermediate.copy()
                    else:
                        # Second segment: intermediate -> end
                        t_elapsed = t - t_intermediate
                        if D_intermediate_to_end == 0:
                            p_traj_dict[t] = p_end.copy()
                        else:
                            #t_seg_total = t_stop_k[k] - t_intermediate 
                            t_seg_total = t_to_end_dur
                            if t_seg_total > 0:
                                next_dir = (t_elapsed / t_seg_total) * (p_end - p_intermediate)
                                p_traj_dict[t] = p_intermediate + next_dir
                            else:
                                p_traj_dict[t] = p_end.copy()
            
            else:  # passive agent (not moving)
                for t in range(T_total):
                    p_traj_dict[t] = p_agents[k].copy()
            
            p_trajectories[k] = p_traj_dict
            phi_trajectories[k] = {t: None for t in range(T_total+1)}
            has_message[k] = {t: False if t < t_intermediate else True for t in range(T_total+1)}

        # Sanity check: verify no agent moves more than sigma per time step
        for k in range(K):
            sigma_tol = 0.1
            if t_stop_k[k] >= 0:
                for t in range(T_total - 1):
                    if t in p_trajectories[k] and (t + 1) in p_trajectories[k]:
                        distance_per_step = la.norm(p_trajectories[k][t + 1] - p_trajectories[k][t])
                        if distance_per_step > sigma + sigma_tol:  # small tolerance for numerical errors
                            raise RuntimeError(
                                f"Agent {k} moved {distance_per_step:.4f} at time step {t}, "
                                f"exceeds sigma={sigma}. "
                                f"Pos t={t}: {p_trajectories[k][t]}, "
                                f"Pos t={t+1}: {p_trajectories[k][t+1]}"
                            )
                        
        p_jammer_trajectory = {t: None for t in range(T_total+1)}
    # END IF isotropic non-jammed
        
    # Non-isotropic or jammer scenarios
    t_pickup_all = {}  # store pickup times for each active agent
    p_jammer_current = None
    v_jammer_current = None
    v_jammer_all = None
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
            p_jammer_trajectory = {}
            p_jammer_trajectory[0] = p_jammer_start # collect all jammer positions at each time point
            v_jammer_all = {}
            v_jammer_all[0] = v_jammer_start  # collect all jammer velocities at each time point

        # Process each agent in the relay chain
        for idx in range(k_D + 1):  # +1 to include retrieving agent
            has_received_message = False
            p_trajectories_k = {}    # {time: np.array(pos_x, pos_y)}
            has_message_k = {}    # {time: bool}

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
            if jammer_on:
                p_handover_max = p_next_relay.copy()
            else:
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
                # Propagate the jammer until the current time step if needed
                if t_current > 0:
                    t_jammer_current = max(p_jammer_trajectory.keys())
                    for t_j in range(t_jammer_current+1, t_current + 1):
                        if not jammer_capsule_boundary(R, Rcom, point=p_jammer_trajectory[t_j - 1]):
                            v_jam_current = -v_jammer_all[t_j - 1]
                        else:
                            v_jam_current = v_jammer_all[t_j - 1]
                        p_jammer_trajectory[t_j] = p_jammer_trajectory[t_j - 1] + v_jam_current
                        v_jammer_all[t_j] = v_jam_current.copy()

                    p_jammer_current = p_jammer_trajectory[t_current].copy()
                    v_jammer_current = v_jammer_all[t_current].copy()
                else:
                    p_jammer_current = p_jammer_trajectory[0].copy()
                    v_jammer_current = v_jammer_all[0].copy()
            
            # While agent has not handed over the message to the next entity
            while t_current <= t_max_w_antenna and not handover_complete:
                p_trajectories_k[t_current] = p_current.copy()
                has_message_k[t_current] = has_received_message
                just_received_message = False

                # Check if current agent has received the message from previous entity (either transmitting base or preceding agent)
                if (t_current > 0) and not has_received_message:
                    if jammer_on and idx == 0:
                        sinr_pickup, _ = calculate_sinr_at_receiver(
                            p_pickup, p_current, phi=None, jammer_pos=p_jammer_current
                        )
                        if sinr_pickup >= 1.0:
                            within_pickup_range = True
                            if idx == 0:  # new pickup for retrieving agent
                                ps_k = p_current.copy()
                        else:
                            within_pickup_range = False
                    else:
                        #within_pickup_range = (la.norm(p_pickup - p_current) < sigma)
                        within_pickup_range = np.isclose(la.norm(p_pickup - p_current), 0.0, atol=1e-5)
                    if within_pickup_range:
                        has_received_message = True
                        t_pickup_all[k_current] = t_current
                        has_message_k[t_current] = has_received_message
                        just_received_message = True

                # Check if communication is possible at current position and antenna direction between the current agent and the next agent
                if has_received_message and not just_received_message:
                    sinr, _ = calculate_sinr_at_receiver(
                        p_current, p_target, phi_current, jammer_pos=p_jammer_current
                    )
                    
                    # Communication is possible if SINR > threshold and within range
                    communication_possible = (sinr >= 1.0)  # hardcoded SNR threshold = 1
                    
                    if communication_possible:  # must be after pickup time
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
                    else:
                        # Reached target position
                        p_current = p_move_target

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
                    p_jammer_trajectory[t_current] = p_jammer_current.copy()
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
    
            if directed_transmission:
                # Adjust steering so that it is performed at the end instead of the beginning (discounting)
                # Find the last phi position and make the beginning steady until that point
                phi_final = antenna_directions[-1]
                
                # Update initial antenna direction until final direction reached   DEBUG: (2*np.pi - phi_start)
                phi_current = phi_start
                n_phi_steps = 0
                phis = []
                #while np.isclose(phi_current, phi_final, atol=1e-2):# and n_phi_steps < 9:
                while not np.isclose(phi_current, phi_final, atol=1e-3) and (n_phi_steps < 9):
                    phi_current = one_antenna_step(phi_current, phi_final, phi_resolution)
                    phis.append(phi_current)
                    n_phi_steps += 1

                if n_phi_steps > 0:
                    n_phi = len(antenna_directions)
                    new_antenna_directions = np.zeros_like(antenna_directions)
                    n_phi_init = n_phi - n_phi_steps
                    new_antenna_directions[:n_phi_init] = [phi_start] * n_phi_init
                    new_antenna_directions[-n_phi_steps:] = phis
                    
                    antenna_directions = new_antenna_directions
                else:  # start and end positions are the same -> no need to steer antenna
                    antenna_directions = phi_start * np.ones_like(antenna_directions) 
            
                # Sanity check, antenna_directions for each time point the agent is active
                if len(antenna_directions) != (t_dur+1):
                    raise RuntimeError("Antenna directions length mismatch with active time steps")

            # If there is a next agent, adjust its relay point 
            if idx < k_D:
                best_relay_points[k_next] = p_next_agent

            # Handle subsequent agent start times due to time difference (save/cut if directed, increase if jammed)
            
            t_stop_dif = t_stop_k[k_current] - t_handover
            t_stop_dif -= np.sign(t_stop_dif)
            t_stop_k[k_current] = t_handover
            for k in best_relaying_agents[idx:]:
                if t_start_k[k] >= t_stop_dif:  # can't have negative start time
                    t_start_k[k] -= (t_stop_dif)
                    t_stop_k[k] -= (t_stop_dif)
                    if t_stop_k[k] < t_handover:
                        t_stop_k[k] = t_handover

                """
                t_stop_dif = max_prev_stop - t_stop_k[k_current]
                for k in best_relaying_agents[idx:]:
                    t_start_k[k] += t_stop_dif
                    t_stop_k[k] += t_stop_dif
                t_stop_k[k_current] = max_prev_stop
                """
                
            """
            t_stop_dif = t_stop_k[k_current] - t_handover
            if t_stop_dif != 0:
                for (i, k) in enumerate(best_relaying_agents[idx:]):
                    t_stop_dif -= np.sign(t_stop_dif) - i
                    if t_start_k[k] >= t_stop_dif:  # can't have negative start time
                        t_start_k[k] -= (t_stop_dif)
                        t_stop_k[k] -= (t_stop_dif)
                        if t_stop_k[k] < t_handover:
                            t_stop_k[k] = t_handover
                t_stop_k[k_current] = t_handover
            """
                    
            # Update agent, distance, timing, and antenna steering
            D_k[k_current] = D_pickup + D_pickup_to_handover
            p_trajectories[k_current] = p_trajectories_k
            phi_trajectories_k = {t: antenna_directions[t - t_start] for t in range(t_start, t_handover + 1)} if directed_transmission else {t: None for t in range(t_start, t_handover + 1)}
            phi_trajectories[k_current] = phi_trajectories_k
            has_message[k_current] = has_message_k

            
        # END FOR each agent in relay chain (including retreiving agent)

        # Trajectories for passive agents
        for k in passive_agents:
            p_trajectories[k] = {0: p_agents[k].copy()}
            phi_trajectories[k] = {0: None}
            has_message[k] = {0: False}

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


    if not no_metrics:
        # Compute budget and metrics (value, delivery time, and total distance), given the trajectories
        #budget = compute_budget(K, R, sigma, beta)
        budget = compute_budget_from_poly(K, R)
    
        value, T_D, D_tot = compute_metrics(p_trajectories, phi_trajectories, c_pos, c_phi, budget, beta)
    else:
        budget = None
        value = None
        T_D = None
        D_tot = None

    if debug:
        print(f"Passive agents: {passive_agents}")

    result = {
        'R': R,
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
        'antenna_directions': phi_trajectories,
        'pickup_times': t_pickup_all,
        'agent_sum_distance': D_tot,
        #'message_air_distance': D_message_air,
        'delivery_time': T_D,
        'value': value,
        'budget': budget,
        'p_jammer_trajectory': p_jammer_trajectory,
        'v_jammer_all': v_jammer_all,
        'ps_k': ps_k,
        'p_trajectories': p_trajectories,
        'phi_trajectories': phi_trajectories,
        'has_message': has_message
    }

    return result


def sample_scenario(K=5, Rcom=1.0, R=5.0, Ra=0.6, sigma=0.2, seed=None):
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


def load_premade_scenario(option=1, Rcom=1.0, sigma=0.2):
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
    elif option == 9:  # test directed transmission timings with K=2
        K = 2
        R = 3*Rcom + 3*sigma
        x = 1.1*R
        p_tx = np.array([0.0, 0.0])
        Ra = 0.6 * R
        p0 = np.array([R/2, 0.0])
        p_recv = np.array([R, 0.0])
        p_agents = np.array([
            [Rcom+sigma, 0.0],
            [2*Rcom+2*sigma, 0.0]
        ])
        phi_agents = None  # or set as needed
        # Dummy jammer (not used in this scenario)
        p_j = np.array([0.0, 0.0])
        dp_j = np.zeros(2)
    elif option == 10:  # test trajectory computation 
        K = 2
        x_dif = 2*Rcom
        R = Rcom + x_dif + sigma
        p_tx = np.array([0.0, 0.0])
        Ra = 0.6 * R
        p0 = np.array([R/2, 0.0])
        p_recv = np.array([R, 0.0])
        p_agents = np.array([
            [Rcom+sigma, 0.0],
            [x_dif+sigma, -sigma]
        ])
        phi_agents = None  # or set as needed
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
    test_scenario = 10

    # Baseline
    clustering = True
    eps = 1e-3

    # Environment parameters
    Rcom = 1.0
    sigma = 0.2
    beta = 0.99

    # Cost parameters
    cost_motion = 0.5
    cost_antenna = 0.1

    # Define scenario
    directed = False
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
        sample = load_premade_scenario(option=test_scenario, Rcom=1.0, sigma=sigma)
        sample['sigma'] = sigma
        
        file_name = fr"test\\test_{test_scenario}_dijkstra_cluster_baseline_seed{seed}"
        animation_save_path = "Animations/test.gif"
        
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
        #animation_save_path = None
        animate_scenario_with_path_colored(sample, result, directed_transmission=directed, 
                                           savepath=animation_save_path, beamer_gif=True)
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