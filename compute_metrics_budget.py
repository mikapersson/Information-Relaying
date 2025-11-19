import numpy as np



def compute_budget_from_poly(K, R, Rcom=1.0):
    """
    Compute budget using polynomial coefficients from a lookup table.

    OBS! Assumes gamma=0.99, sigma=0.2, and Rcom=1.
    
    Args:
        K (int): Number of agents
        R (float): Distance between transmitter and receiver
        Rcom (float): Communication range (default: 1.0)
    
    Returns:
        float: The computed budget(w) using polynomial approximation
    
    Raises:
        ValueError: If R is outside the valid range [K*Rcom, (K+4)*Rcom]
        KeyError: If K is not found in the lookup table
    """

    # Hardcoded polynomial coefficients (see budget_evaluation_poly.py)
    hardcoded_coeffs = {
        1: [0.008064426708540176, 0.24227644166893686, 0.447558811375325],
        2: [0.008563631604393761, 0.2630189822526008, 0.5842598858548388],
        3: [0.00916296576633421, 0.28665157865366997, 0.9336890390584559],
        4: [0.009798427621887198, 0.3060525869549174, 1.5485362284764004],
        5: [0.010414558965496229, 0.33280010372979313, 2.382222977668543],
        6: [0.011373063174744277, 0.3470956191782513, 3.570526510127933],
        7: [0.011840565900633405, 0.38027890499105743, 4.943145118749329],
        8: [0.012999363841415444, 0.38982808982575157, 6.8011931110736885],
        9: [0.013465327379084442, 0.42848574257521205, 8.788865616040736],
        10: [0.01494447983453105, 0.42915282929532317, 11.451549205015272]
    }
    
    if K not in hardcoded_coeffs:
        raise KeyError(f"Coefficients for K={K} not found in lookup table. Available K values: {list(hardcoded_coeffs.keys())}")
    
    # Validate R is within the valid range
    R_min = K * Rcom
    R_max = (K + 4) * Rcom
    
    if R < R_min or R > R_max:
        raise ValueError(
            f"R={R} is outside the valid range [{R_min}, {R_max}] for K={K}. "
            f"The polynomial coefficients are only valid within this interval."
        )
    
    coeffs = hardcoded_coeffs[K]
    poly = np.poly1d(coeffs)
    return float(poly(R))


def compute_metrics(p_trajectories, phi_trajectories, c_pos, c_phi, budget, beta):
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
    
    # Compute total distance traveled by all agents
    D_tot = 0.0
    for k in range(K):
        if k in p_trajectories:
            p_traj_k = p_trajectories[k]
            for t in range(len(p_traj_k)-1):
                if t in p_traj_k and (t + 1) in p_traj_k:
                    distance_step = np.linalg.norm(p_traj_k[t + 1] - p_traj_k[t])
                    D_tot += distance_step
    
    # Compute antenna steering cost
    phi_cost_total = 0.0
    if c_phi > 0:
        for k in range(K):
            if k in phi_trajectories:
                phi_traj_k = phi_trajectories[k]
                
                # Sum antenna steering costs (only when antenna direction changes)
                times_k = list(phi_traj_k.keys())
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
                p_diff = np.linalg.norm(p_next - p_current) if p_current is not None and p_next is not None else 0.0
                
                # Only count cost if antenna actually moved
                if p_current is not None and p_next is not None:
                    position_cost += beta**t * c_pos * p_diff**2  # Cost per antenna step
    
    # Total cost
    total_cost = position_cost + phi_cost_total
    
    # Compute value metric
    # Value = budget - total_cost (higher is better)
    value = (beta**T_D)*budget - total_cost
    
    return value, T_D, D_tot


if __name__ == "__main__":
    # Example usage
    K = 2
    R = 4.0
    budget = compute_budget_from_poly(K, R)
    print(f"Computed budget for K={K}, R={R}: {budget}")

    # Toy trajectory (positions and antenna orientations over each time step for each agent)
    p_trajectories = {0: {0: np.array([ 4.26894408, -0.75066582]), 1: np.array([ 4.26894408, -0.75066582]), 2: np.array([ 4.26894408, -0.75066582]), 3: np.array([ 4.26894408, -0.75066582]), 4: np.array([ 4.26894408, -0.75066582]), 5: np.array([ 4.11738685, -0.63901587]), 6: np.array([ 3.96582962, -0.52736592]), 7: np.array([ 3.8142724 , -0.41571597]), 8: np.array([ 3.66271517, -0.30406602]), 9: np.array([ 3.51115794, -0.19241607]), 10: np.array([ 3.35960071, -0.08076612]), 11: np.array([3.20804348, 0.03088383]), 12: np.array([3.05648626, 0.14253378]), 13: np.array([3.05747607, 0.14239139])}, 
            1: {0: np.array([1.67438549, 0.85330348]), 1: np.array([1.51770274, 0.77345452]), 2: np.array([1.36102   , 0.69360557]), 3: np.array([1.20433726, 0.61375661]), 4: np.array([1.04765451, 0.53390766]), 5: np.array([0.89097177, 0.45405871]), 6: np.array([1.08692244, 0.42586978]), 7: np.array([1.28287312, 0.39768086]), 8: np.array([1.4788238 , 0.36949194]), 9: np.array([1.67477447, 0.34130302]), 10: np.array([1.87072515, 0.31311409]), 11: np.array([2.06667582, 0.28492517]), 12: np.array([2.06667582, 0.28492517]), 13: np.array([2.06667582, 0.28492517])}}
    
    phi_trajectories = {0: {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None}, 
                        1: {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None}}

    # Compute and print metrics
    value, T_D, D_tot = compute_metrics(p_trajectories, phi_trajectories, 1.0, 0.5, budget, 0.99)
    print(f"Metrics - Value: {value}, Delivery Time: {T_D}, Total Distance: {D_tot}")