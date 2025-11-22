import numpy as np
import matplotlib.pyplot as plt

def compute_v_sharp(R, R_com, sigma, beta, c_delta_p):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    V = (beta ** t_max) - c_delta_p * sigma ** 2 * ((1 - beta ** (t_max + 1)) / (1 - beta))
    return V, t_max

def conservative_upper_bound(R, R_com, sigma, beta):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    numerator = (1 - beta) * (beta ** t_max)
    denominator = sigma ** 2 * (1 - beta ** (t_max + 1))
    return numerator / denominator

def plot_cost_vs_R():
    # Fixed parameters
    R_com = 0.2
    sigma = 0.1
    beta = 0.95
    R_values = np.linspace(0.1, 2.0, 200)

    upper_bounds = []
    V_values = []

    # Fixed cost to test
    test_c_delta_p = 0.1

    for R in R_values:
        bound = conservative_upper_bound(R, R_com, sigma, beta)
        upper_bounds.append(bound)
        V, _ = compute_v_sharp(R, R_com, sigma, beta, test_c_delta_p)
        V_values.append(V)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('R (Environment Radius)')
    ax1.set_ylabel('Upper Bound on $c_{\delta p}$', color=color1)
    ax1.plot(R_values, upper_bounds, label='Conservative Upper Bound', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=test_c_delta_p, color='gray', linestyle='--', label=f'Test $c_{{\delta p}}$ = {test_c_delta_p}')

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Value $V(s^\\sharp)$', color=color2)
    ax2.plot(R_values, V_values, label='Value at test $c_{\delta p}$', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='black', linestyle=':')

    # Legends and title
    fig.tight_layout()
    plt.title('Conservative Cost Bound and Value vs R')
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_cost_vs_R()
