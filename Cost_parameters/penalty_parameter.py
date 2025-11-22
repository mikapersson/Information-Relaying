import numpy as np

# B.1: Conservative range of cost of motion c_{\delta p}

def compute_v_sharp(R, R_com, sigma, beta, c_delta_p):  # Computes the value function at s^\sharp for scenario B.1
    # Compute minimal distance
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))  # time it takes for the solo agent

    # Compute value function
    V = (beta ** t_max) - c_delta_p * sigma ** 2 * ((1 - beta ** (t_max + 1)) / (1 - beta))
    return V, t_max

def conservative_upper_bound(R, R_com, sigma, beta):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    numerator = 5*(1 - beta) * (beta ** t_max)
    denominator = sigma ** 2 * (1 - beta ** (t_max + 1))
    return numerator / denominator, t_max

def experiment_with_cdelta_p():
    # Constants 
    R_com = 1.0          # Max communication radius  (given by SNR=SNR_0=1)
    K = 5                # Number of agents
    R_max = (K+4)*R_com  # Assuming one player
    R = R_max            # R_max
    sigma = 0.1            # Max speed
    beta = 0.99          # Discount factor

    upper_bound, t_max = conservative_upper_bound(R, R_com, sigma, beta)
    print(f"Conservative upper bound for c_delta_p: {upper_bound:.6f} (t_max = {t_max})")

    # Try different values of c_delta_p
    test_values = np.linspace(0.0, upper_bound * 1.5, 10)

    print("\nTesting different c_delta_p values:")
    for c in test_values:
        V, _ = compute_v_sharp(R, R_com, sigma, beta, c)
        print(f"c_delta_p = {c:.6f} --> V(s^#) = {V:.6f} {'<-- POSITIVE' if V > 0 else ''}")

if __name__ == "__main__":
    experiment_with_cdelta_p()
