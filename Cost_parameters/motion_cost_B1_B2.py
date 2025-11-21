import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Conservative
# -------------------------
def compute_v_sharp(R, R_com, sigma, beta, c_delta_p):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    V = (beta ** t_max) - c_delta_p * sigma ** 2 * ((1 - beta ** (t_max + 1)) / (1 - beta))
    return V

def conservative_upper_bound(R, R_com, sigma, beta):
    D_min = 2.1 * R - 3 * R_com
    t_max = int(np.ceil(D_min / sigma))
    numerator = (1 - beta) * (beta ** t_max)
    denominator = sigma ** 2 * (1 - beta ** (t_max + 1))
    return numerator / denominator

# -------------------------
# Less conservative
# -------------------------
def compute_value_chain_policy(K, R_com, sigma, beta, c_delta_p):
    R = (K + 4) * R_com
    D1 = 1.1 * R + 2 * R_com
    t1_2 = int(np.ceil(D1 / sigma))
    t_max = t1_2 + K 

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

    value = beta ** t_max
    motion_cost = 0.0
    for n in range(t_max + 1):
        for k in range(K):
            if t_k_1[k] <= n <= t_k_2[k]:
                motion_cost += beta ** n

    value -= c_delta_p * sigma ** 2 * motion_cost
    return value

# -------------------------
# Comparison
# -------------------------
def compare_value_functions():
    # Shared parameters
    K = 5
    R_com = 1
    R = (K + 4) * R_com
    sigma = 0.1
    beta = 0.99

    # Conservative bound
    upper_bound = conservative_upper_bound(R, R_com, sigma, beta)

    # Range of c_delta_p
    c_values = np.linspace(0.0, 0.3, 200)  # 0.3 = upper_bound * 1.5

    V_cons = []
    V_less = []
    diffs = []
    ratios = []

    for c in c_values:
        v_cons = compute_v_sharp(R, R_com, sigma, beta, c)
        v_less = compute_value_chain_policy(K, R_com, sigma, beta, c)

        V_cons.append(v_cons)
        V_less.append(v_less)
        diffs.append(v_less - v_cons)

        # ratio handling
        if abs(v_cons) > 1e-8:  # avoid division by 0
            ratios.append(v_less / v_cons)
        else:
            ratios.append(np.nan)

    V_cons = np.array(V_cons)
    V_less = np.array(V_less)
    diffs = np.array(diffs)
    ratios = np.array(ratios)

    # -------------------------
    # Figure with 3 subplots
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

    # Top subplot: overlay values
    ax1.plot(c_values, V_cons, label=r"$V_{\mathrm{cons}}$", color="blue")
    ax1.plot(c_values, V_less, label=r"$V_{lesscons}$", color="green")
    #ax1.axhline(0, color="red", linewidth=1.6, label=r"$V_{passive}$")
    ax1.plot(c_values, np.zeros_like(c_values), label=r"$V_{passive}$", color="red")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Value functions")
    ax1.set_title(fr"Value Functions vs $c_{{\delta p}}$ "
                  fr"($\sigma={sigma}$, $R_{{com}}={R_com}$, $K={K}$, $\beta={beta}$)")
    ax1.legend()
    ax1.grid(True)

    # Middle subplot: V_conservative - V_less_conservative
    ax2.plot(c_values, diffs, color="blue", label=r"$V_{lesscons} - V_{cons}$")
    #ax2.axhline(0, color="red", linestyle="--", linewidth=1.2, label="equal value")
    ax2.plot(c_values, -V_cons, color="red", label=r"$V_{passive} - V_{cons}$")
    ax2.fill_between(c_values, diffs, 0, where=diffs > 0, color="green", alpha=0.2,
                     label="less conservative best")
    ax2.fill_between(c_values, diffs, 0, where=(V_cons > V_less) & (V_cons > 0), color="blue", alpha=0.2,
                     label="conservative best")
    ax2.fill_between(c_values, -V_cons, 0, where=-V_cons > 0, color="red", alpha=0.2,
                     label="passive best")
    ax2.set_ylabel("Value difference")
    ax2.set_title(r"Comparing value functions")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-0.05, 0.11)
    

    # Bottom subplot: V_passive - V_conservative
    """
    ax3.plot(c_values, ratios, color="blue", label=r"$V_{lesscons} / V_{cons}$")
    ax3.axhline(1, color="red", linestyle="--", linewidth=1.2, label="equal ratio")
    ax3.set_xlabel(r"$c_{\delta p}$")
    ax3.set_ylabel("Value ratio")
    ax3.set_title("Ratio of value functions")
    ax3.legend()
    ax3.grid(True)
    """

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_value_functions()