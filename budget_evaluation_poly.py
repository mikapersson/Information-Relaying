import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def fit_polynomial_coefficients(K, sigma=0.1, gamma=0.99):
    """
    Fit second-order polynomial coefficients for budget as a function of R.
    
    Args:
        K (int): Number of agents
        sigma (float): Displacement size
        gamma (float): Discount factor
    
    Returns:
        np.ndarray: Polynomial coefficients [a, b, c] for budget = a*R^2 + b*R + c
    """
    Rcom = 1.0
    Rmin = K * Rcom
    Rmax = (K + 4) * Rcom
    R_values = np.linspace(Rmin, Rmax, 1000)
    budget_values = np.array([compute_budget(K, R, sigma, gamma) for R in R_values])
    
    # Fit second-order polynomial
    coeffs = np.polyfit(R_values, budget_values, 2)
    return coeffs


def regenerate_polynomial_coefficients(sigma=0.2, gamma=0.99, save=True):
    """
    Regenerate polynomial coefficients fitted to the corrected compute_budget function.
    
    Args:
        sigma (float): Displacement size
        gamma (float): Discount factor
        save (bool): Whether to save coefficients to file
    
    Returns:
        dict: Dictionary of polynomial coefficients {K: [a, b, c]}
    """
    coeffs_dict = {}
    
    print("Regenerating polynomial coefficients...")
    print("=" * 100)
    print(f"{'K':<5} {'a':<20} {'b':<20} {'c':<20}")
    print("=" * 100)
    
    for K in range(1, 11):
        Rcom = 1.0
        Rmin = K * Rcom
        Rmax = (K + 4) * Rcom
        R_values = np.linspace(Rmin, Rmax, 1000)
        budget_values = np.array([compute_budget(K, R, sigma, gamma) for R in R_values])
        
        # Fit second-order polynomial
        coeffs = np.polyfit(R_values, budget_values, 2)
        coeffs_dict[K] = coeffs
        
        print(f"{K:<5} {coeffs[0]:<20.15e} {coeffs[1]:<20.15e} {coeffs[2]:<20.15e}")
    
    print("=" * 100)
    
    if save:
        save_polynomial_coefficients(coeffs_dict, "polynomial_coefficients_sigma02.json")
    
    return coeffs_dict


def update_hardcoded_coefficients(coeffs_dict):
    """
    Update the hardcoded coefficients in compute_budget_from_poly.
    Print code ready to copy-paste.
    
    Args:
        coeffs_dict (dict): Dictionary of polynomial coefficients
    """
    print("\n" + "=" * 100)
    print("Copy-paste the following into compute_budget_from_poly():")
    print("=" * 100 + "\n")
    print("    hardcoded_coeffs = {")
    
    for K in sorted(coeffs_dict.keys()):
        coeffs = coeffs_dict[K]
        print(f"        {K}: [{coeffs[0]}, {coeffs[1]}, {coeffs[2]}],")
    
    print("    }")
    print("\n" + "=" * 100)


def compute_budget_from_poly(K, R, Rcom=1.0):
    """
    Compute budget using polynomial coefficients from a lookup table.
    
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
    # Hardcoded polynomial coefficients
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


def save_polynomial_coefficients(coeffs_dict, save_path="polynomial_coefficients.json"):
    """
    Save polynomial coefficients to a JSON file.
    
    Args:
        coeffs_dict (dict): Dictionary of coefficients {K: [a, b, c]}
        save_path (str): Path to save the JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    coeffs_dict_serializable = {str(k): v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in coeffs_dict.items()}
    
    with open(save_path, 'w') as f:
        json.dump(coeffs_dict_serializable, f, indent=4)
    print(f"Polynomial coefficients saved to {save_path}")

def compute_budget(K, R, sigma, gamma):
    """
    Compute the budget(w) as described in the prompt.

    Args:
        K (int): Number of agents
        R (float): Distance between transmitter and receiver
        sigma (float): Displacement size
        gamma (float): Discount factor

    Returns:
        float: The computed budget(w)
    """
    Rcom = 1.0

    # Compute T_sharp
    T_sharp = int(np.floor((1.1 * R + 2 * Rcom) / sigma) + K)

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

    # Compute budget(w)
    budget = 0.0
    for t in range(T_sharp):
        gamma_t = gamma ** t
        active_agents = sum(
            t_start[k] <= t < t_stop[k] for k in range(K)
        )
        budget += gamma_t * sigma**2 * active_agents
    budget = budget / (gamma ** T_sharp)

    return budget

def main(linewidth=2.0, tick_fontsize=10, save_plot=True, plot_dir="Plots", 
         plot_type="both"):
    """
    Plot budget evaluation for different numbers of agents using hardcoded polynomial coefficients.
    
    Args:
        linewidth (float): Thickness of the lines in the plot
        tick_fontsize (int): Font size for tick labels
        save_plot (bool): Whether to save the plot as PDF
        plot_dir (str): Directory to save the plot
        plot_type (str): "data" (original data), "fit" (polynomial fit), or "both" (both curves)
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Set LaTeX rendering
    font_size = 20
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': font_size-5,
    })
    
    sigma = 0.2
    gamma = 0.99
    Rcom = 1.0
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # Get viridis colormap
    cmap = plt.cm.viridis
    n_K = 9
    colors = [cmap(i / (n_K - 1)) for i in range(n_K)]
    colors[8] = 'orange'  # Make K=9 orange
    
    for K in range(1, n_K+1):
        Rmin = K * Rcom
        Rmax = (K + 4) * Rcom
        R_values = np.linspace(Rmin, Rmax, 1000)
        budget_values = []
        
        for R in R_values:
            budget = compute_budget(K, R, sigma, gamma)
            budget_values.append(budget)
        
        budget_values = np.array(budget_values)
        
        # Plot original data
        if plot_type in ["data", "both"]:
            ax.plot(R_values, budget_values, label=f'${K}$', 
                    linewidth=linewidth, color=colors[K-1], linestyle='-')
        
        # Plot polynomial fit using hardcoded coefficients from compute_budget_from_poly
        if plot_type in ["fit", "both"]:
            fit_values = np.array([compute_budget_from_poly(K, R) for R in R_values])
            
            linestyle = '--' if plot_type == "both" else '-'
            alpha_val = 0.7 if plot_type == "both" else 1.0
            label_suffix = ' (fit)' if plot_type == "both" else ''
            
            ax.plot(R_values, fit_values, label=f'${K}${label_suffix}', 
                    linewidth=linewidth, color=colors[K-1], linestyle=linestyle, alpha=alpha_val)
    
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$\mathrm{budget}$')
    ax.legend(loc='upper left', title=r'$K$', reverse=True)
    ax.grid(False, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    
    # Save as PDF if requested
    if save_plot:
        os.makedirs(plot_dir, exist_ok=True)
        filename = f'budget_evaluation_{plot_type}_poly.pdf'
        plot_path = os.path.join(plot_dir, filename)
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Budget plot saved to {plot_path}")
    
    plt.show()


#if __name__ == "__main__":
#    main(linewidth=2.5, tick_fontsize=16, save_plot=True, plot_dir="Plots", 
#         plot_type="both")
    
if __name__ == "__main__":
    # Step 1: Generate new polynomial coefficients for sigma=0.2
    print("Step 1: Generating polynomial coefficients for sigma=0.2...\n")
    
    coeffs_dict = {}
    print("=" * 100)
    print(f"{'K':<5} {'a':<20} {'b':<20} {'c':<20}")
    print("=" * 100)
    
    for K in range(1, 21):
        coeffs = fit_polynomial_coefficients(K, sigma=0.2, gamma=0.99)
        coeffs_dict[K] = coeffs
        print(f"{K:<5} {coeffs[0]:<20.15e} {coeffs[1]:<20.15e} {coeffs[2]:<20.15e}")
    
    print("=" * 100)
    
    # Step 2: Save coefficients to JSON file
    print("\nStep 2: Saving coefficients to file...")
    #save_polynomial_coefficients(coeffs_dict, "polynomial_coefficients_sigma02.json")
    
    # Step 3: Print code ready to copy-paste
    print("\nStep 3: Generated code to copy-paste into compute_budget_from_poly()...")
    update_hardcoded_coefficients(coeffs_dict)
    
    # Step 4: Plot comparison
    print("\nStep 4: Plotting comparison...\n")
    main(linewidth=2.5, tick_fontsize=16, save_plot=True, plot_dir="Media/Figures", 
         plot_type="fit")