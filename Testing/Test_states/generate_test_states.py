import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


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

def plot_capsule_samples(R=10.0, Rcom=1.0, n_samples=1000, seed=0):
    rng = np.random.default_rng(100)
    samples = np.array([sample_capsule_point(R, Rcom, rng) for _ in range(n_samples)])

    # Capsule outline for reference
    height = 3 * Rcom
    radius = 1.5 * Rcom

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label='Samples')

    # Draw rectangle (left edge at x=0, width=R)
    rect = plt.Rectangle((0, -height/2), R, height, edgecolor='k', facecolor='none', lw=2)
    ax.add_patch(rect)
    # Draw left half circle (centered at (0,0))
    circle_left = plt.Circle((0, 0), radius, edgecolor='k', facecolor='none', lw=2)
    ax.add_patch(circle_left)
    # Draw right half circle (centered at (R,0))
    circle_right = plt.Circle((R, 0), radius, edgecolor='k', facecolor='none', lw=2)
    ax.add_patch(circle_right)

    ax.set_aspect('equal')
    ax.set_title("Samples from sample_capsule_point")
    ax.legend()
    plt.show()

def generate_test_states(
    nr_eval=10000,
    K=5,
    Rcom=1.0,
    Ra_factor=0.6,
    agent_sigma=0.2,
    jammer_sigma=0.1,
    beta=0.99,
    seed=42,
    save_path="testing_states.csv"
):
    rng = np.random.default_rng(seed)
    records = []

    R_min = K * Rcom
    R_max = (K + 4) * Rcom

    for i in range(nr_eval):
        # Sample R for this scenario
        R = rng.uniform(R_min, R_max)
        #R = R_max
        Ra = Ra_factor * R

        # Transmitting base at origin
        p_tx = np.array([0.0, 0.0])

        # Receiver base
        p_rx = np.array([R, 0.0])

        # Midpoint for agents initialization
        p0 = np.array([R/2, 0.0])

        # Agents positions around p0
        agent_radiuses = Ra * np.sqrt(rng.uniform(0, 1, size=(K, 1)))
        agent_angles = rng.uniform(0, 2*np.pi, size=(K, 1))
        p_agents = p0 + agent_radiuses * np.hstack([np.cos(agent_angles), np.sin(agent_angles)])
        
        # Antenna orientations for each agent (except jammer)
        antenna_orientations = rng.uniform(0, 2*np.pi, size=K)

        # Jammer position
        jammer_pos = sample_capsule_point(R, Rcom, rng)

        # Jammer displacement (velocity vector)
        # Uniformly distributed in the half plane directed from jammer_pos towards the midpoint between the bases
        pmid = (p_tx + p_rx) / 2
        theta_j_prime = np.arctan2(pmid[1] - jammer_pos[1], pmid[0] - jammer_pos[0])
        theta_j = rng.uniform(theta_j_prime - np.pi/2, theta_j_prime + np.pi/2)
        jammer_disp = jammer_sigma * np.array([np.cos(theta_j), np.sin(theta_j)])  # assume max velocity
        
        # Save as a flat record
        record = {
            'idx': i+1,  
            'R': R,
            'Rcom': Rcom,
            'Ra': Ra,
            'p_tx_x': p_tx[0],
            'p_tx_y': p_tx[1],
            'p_rx_x': p_rx[0],
            'p_rx_y': p_rx[1],
            'jammer_x': jammer_pos[0],
            'jammer_y': jammer_pos[1],
            'jammer_dx': jammer_disp[0],
            'jammer_dy': jammer_disp[1]
        }
        # Add agent positions and orientations
        for k in range(K):
            record[f'agent{k+1}_x'] = p_agents[k, 0]
            record[f'agent{k+1}_y'] = p_agents[k, 1]
            record[f'agent{k+1}_phi'] = antenna_orientations[k]
        record[f'agent_sigma'] = agent_sigma
        record[f'jammer_sigma'] = jammer_sigma
        record[f'beta'] = beta
        record[f'seed'] = seed
        record[f'K'] = K
        records.append(record)  


    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"Saved {nr_eval} testing states to {save_path}")


if __name__ == "__main__":
    nr_eval=10000
    K=10
    Rcom=1.0
    Ra_factor = 0.6  # later multiplied with R
    seed=2
    agent_sigma = 0.2
    jammer_sigma = 0.1
    beta = 0.99

    # plot_capsule_samples(R=(K+4)*Rcom, Rcom=1.0, n_samples=1000, seed=0)
    for k in range(1,K+1):
        generate_test_states(
            nr_eval=nr_eval,
            K=k,
            Ra_factor=Ra_factor,
            Rcom=Rcom,
            agent_sigma=agent_sigma,
            jammer_sigma=jammer_sigma,
            beta=beta,
            seed=seed,
            save_path=f"Testing/Test_states/test_states_K{k}_n{nr_eval}.csv"
        )
