import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import matplotlib.cm as cm

# -------------------------
# Compute value function for k_active agents
# -------------------------
def compute_value_k_active(K, k_active, R_com, sigma, beta, c_delta_p, cphi):
    """
    Compute V_k(s^sharp) for k_active agents using your LaTeX definitions.
    """
    R = (K + 4) * R_com

    # Distances D_{k,kact}
    D_k = np.zeros(k_active)
    for k in range(1, k_active + 1):
        if k == 1:
            D_k[0] = 1.1 * R + (K - k_active + 2) * R_com
        else:
            D_k[k - 1] = 0.1 * R + (k_active - k + 1) * R_com

    # Diagnostics
    D_tot = sum(D_k)
    #print(f"Total distance Dk for {k_active} agents: {sum(D_k)}")


    # Start and stop times t^1_{k,kact} and t^2_{k,kact}
    t_k_1 = np.zeros(k_active, dtype=int)
    t_k_2 = np.zeros(k_active, dtype=int)

    t_k_1[0] = 0
    t_k_2[0] = int(np.ceil(D_k[0] / sigma))
    t_max = t_k_2[0] + k_active

    for k in range(2, k_active + 1):
        t_k_1[k - 1] = int(np.floor((D_k[0] - D_k[k - 1]) / sigma)) + (k - 2)
        t_k_2[k - 1] = t_max - (K - (k_active - k + 1))

    # Compute value: start with delivery reward 
    #V = R*beta ** t_max  # reward at t_max
    #V = D_tot*beta ** t_max  # reward at t_max
    #V = D_tot/5  # reward at t_max
    #V = (1-beta**(t_max+1))/(1-beta)*(D_tot/t_max)  # 2025-09-22
    #V = (1-beta**(t_max))/(1-beta)*(D_tot/t_max)**2  # USE THIS 2025-10-16  14-14

    #V = (1-beta**(t_max))/(1-beta) * (D_tot**2/(K*t_max**2))  # 2025-10-20  most reasonable yet?
    #V = 1/(1-beta) * (D_tot**2/(K*t_max**2))  # 2025-10-20
    #V = 1/(1-beta**t_max) * (D_tot**2/(K*t_max**2))  # 2025-10-20
    #V = ((1-beta**t_max) * D_tot**2) / (K * (1-beta) * beta**t_max * t_max**2)  # 2025-10-20
    #V = (1-beta**(t_max))/(1-beta) * (D_tot**2/(K * beta**t_max * t_max**2))  # 2025-10-20  not good but most motivated
    

    #V = (1-beta**(t_max))/(1-beta) * (D_tot**2/(K * beta**(t_max) * t_max**2))  # 2025-10-21  Helt enkelt använd V(s#)

    #V = (1-beta**(t_max+1))/(1-beta)*((D_tot*sigma**2 + (np.pi/8)**2*8*K)/t_max)

    # Compute costs
    motion_cost = 0.0
    antenna_cost = 0.0

    for t in range(t_max + 1):
        for k in range(k_active):
            # Motion cost
            if t_k_1[k] <= t <= t_k_2[k]:
                motion_cost += beta ** t
            # Antenna cost: last 8 steps before stopping
            if t_k_2[k] - 7 <= t <= t_k_2[k]:
                antenna_cost += beta ** t

    # Costs without parameters yet
    motion_cost = motion_cost * sigma ** 2  
    antenna_cost = antenna_cost * (np.pi / 8) ** 2

    # Compute budget based on total movement cost in s#
    budget = motion_cost / beta ** t_max

    V = beta**t_max * budget - c_delta_p * motion_cost - cphi * antenna_cost

    return V

# -------------------------
# Compute values for all k_active
# -------------------------
def compute_all_values(K, R_com, sigma, beta, c_delta_p, cphi):
    values = []
    for k_active in range(1, K + 1):
        val = compute_value_k_active(K, k_active, R_com, sigma, beta, c_delta_p, cphi)
        values.append(val)
    return np.array(values)

# -------------------------
# Comparison and 3D plot
# -------------------------
def compare_value_functions_3d():
    # Parameters
    K_min = 1
    K_max = 30
    step = 1
    R_com = 1.0
    sigma = 0.2
    beta = 0.99

    motion_max_cost = 1.5
    antenna_max_cost = 1.5
    N = 50  # grid points per axis

    for K in range(K_min, K_max+1, step):
        motion_cost_range = np.linspace(0.0, motion_max_cost, N)
        antenna_cost_range = np.linspace(0.0, antenna_max_cost, N)

        # For level curves plotting
        N = len(motion_cost_range)
        M = len(antenna_cost_range)
        V_all_k_grid = np.zeros((N, M, K))  # shape: (motion_idx, antenna_idx, k_active)

        motion_vals, antenna_vals = np.meshgrid(motion_cost_range, antenna_cost_range)
        best_policy = np.zeros_like(motion_vals, dtype=int)  # index of best policy for each cost pair

        # Compute values for all policies
        for i in range(N):
            for j in range(N):
                c_delta_p = motion_cost_range[i]
                cphi = antenna_cost_range[j]

                # Compute V_k for k=1..K
                V_all_k = compute_all_values(K, R_com, sigma, beta, c_delta_p, cphi)

                V_all_k_grid[i, j, :] = V_all_k

                # Passive policy = 0
                V_best = np.max(np.append(V_all_k, 0))
                if V_best == 0:
                    best_policy[i, j] = 0  # Passive
                else:
                    best_k = np.argmax(V_all_k) + 1
                    best_policy[i, j] = best_k  # Best number of active agents


        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        if K < 0:  # initially K==5
            passive_color = "black"
            colors = [passive_color] + ['blue', 'orange', 'green', 'red', 'purple'][:K]  # adjust length to K
        else:
            # Get K colors from viridis colormap
            viridis_colors = [cm.viridis(i / (K - 1)) for i in range(K)] if K > 1 else [cm.viridis(0.5)]

            # Prepend black to the colormap
            passive_color = "red"
            colors = [passive_color] + viridis_colors
        cmap = ListedColormap(colors)    
        ax.pcolormesh(motion_vals, antenna_vals, best_policy.T, cmap=cmap, alpha=0.8)

        # Loop over all k_active to plot their contours/level curves
        for k in range(K):
            V_k_grid = V_all_k_grid[:, :, k]  # pick the k-th agent value function

            # Mask values outside the best-policy region for this k
            mask = (best_policy != (k + 1))  # True where this k is NOT best
            V_k_masked = np.array(V_k_grid, copy=True)
            V_k_masked[mask] = np.nan

            # Only contour within best-policy region
            CS = ax.contour(
                motion_vals, antenna_vals, V_k_masked.T,  # note transpose for correct orientation
                levels=10,  # number of contour levels
                #colors=f'C{k}',  # color for this agent
                colors="black",  # color for this agent
                linewidths=1
            )
            #ax.clabel(CS, inline=True, fontsize=8, fmt=f'k={k+1}: %.2f')


        ax.set_xlabel('Motion cost $c_{\\delta p}$')
        ax.set_ylabel('Antenna cost $c_{\\phi}$')
        ax.set_title(fr'Best Policy over costs ($K=${K}, $R_{{\mathrm{{com}}}}=${R_com}, $\sigma=${sigma}, $\beta=${beta})')

        # Legend
        if K > 10:
            # Save the pcolormesh object so we can use it in colorbar
            c = ax.pcolormesh(motion_vals, antenna_vals, best_policy.T, cmap=cmap, alpha=0.8)
            proxy_passive = mlines.Line2D([0], [0], color=passive_color, lw=4, label='Passive')
            cbar = plt.colorbar(c, ax=ax)
            cbar.set_label('Number of Active Agents', rotation=270, labelpad=15)
            ax.legend(handles=[proxy_passive], loc='upper right')
        else:
            proxy_passive = mlines.Line2D([0], [0], color=passive_color, lw=4, label='Passive')
            proxy_actives = [mlines.Line2D([0],[0],color=colors[k+1],lw=4,label=f'{k+1} active') for k in range(K)]
            proxy_contour = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='contours')
            ax.legend(handles=[proxy_passive]+proxy_actives+[proxy_contour], loc='upper right')

        plt.tight_layout()    
        plt.savefig("Media\\Figures\\motion_antenna_cost_compare_{}_agents.png".format(K), dpi=1000, bbox_inches="tight")
        print("Plot saved for K={}".format(K))
        #plt.show()

if __name__ == "__main__":
    compare_value_functions_3d()
