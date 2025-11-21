import numpy as np

# B.2: A less conservative range of cost of motion c_{\delta p}

def compute_value_chain_policy(K, R_com, sigma, beta, c_delta_p):
    """
    Compute the value function V(s^) for the full chain policy with K agents.
    """
    R = (K + 4) * R_com
    D1 = 1.1 * R + 2 * R_com
    t1_2 = int(np.ceil(D1 / sigma))
    t_max = t1_2 + K 

    # Time windows for each agent's movement
    t_k_1 = np.zeros(K, dtype=int)
    t_k_2 = np.zeros(K, dtype=int)
    for k in range(1, K + 1):
        if k == 1:
            t_k_1[k - 1] = 0
            t_k_2[k - 1] = t1_2
        else:
            Dk = 1.1 * R - (3 + k) * R_com
            start_time = int(np.floor((D1 - Dk) / sigma)) + (k - 2)
            stop_time = t_max - (K - k + 1)
            t_k_1[k - 1] = start_time
            t_k_2[k - 1] = stop_time

    # Total discounted reward and motion cost
    value = K*beta ** t_max
    motion_cost = 0.0

    for n in range(t_max + 1):
        for k in range(K):
            if t_k_1[k] <= n <= t_k_2[k]:
                motion_cost += beta ** n

    value -= c_delta_p * sigma ** 2 * motion_cost
    return value, t_max, t_k_1, t_k_2


def experiment_with_cdelta_p_full_chain():
    # Configuration
    R_com = 1         # Communication range
    K = 5             # Number of agents
    sigma = 0.1       # Max speed
    beta = 0.99       # Discount factor

    # Try various c_delta_p values
    test_c_values = np.linspace(1.176, 1.177, 20)  # K=5: 0.45, 0.58 -> 0.235

    print(f"Experimenting with full-chain policy (K={K} agents):\n")
    for c in test_c_values:
        V, t_max, t1s, t2s = compute_value_chain_policy(K, R_com, sigma, beta, c)
        status = "POSITIVE" if V > 0 else "NEGATIVE"
        print(f"c_delta_p = {c:.4f} --> V(s^#) = {V:.6f} [{status}]")

    print(f"t_max = {t_max}")

    print("\nAgent active intervals:")
    for k in range(K):
        print(f"  Agent {k+1}: [{t1s[k]}, {t2s[k]}]")

if __name__ == "__main__":
    experiment_with_cdelta_p_full_chain()
