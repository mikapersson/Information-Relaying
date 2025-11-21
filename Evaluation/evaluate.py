import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image



# Add the parent directory to sys.path to import baseline
sys.path.append(str(Path(__file__).parent.parent))
from Baseline.baseline import baseline, plot_scenario_with_path_colored, plot_scenario

from Baseline.communication import communication_range

import matplotlib.pyplot as plt

def get_state_dict(data, row_idx):
    """
    Reads the specified row from the dataframe and returns a dictionary with scenario parameters.
    Args:
        data: pandas DataFrame containing the scenario data.
        row_idx: integer index of the row to read.
    Returns:
        dict with keys: K, p_agents, p_tx, p_rx, Rcom, sigma, beta
    """
    try:
        state = data.iloc[row_idx]
        
        # Debug: Print available columns
        if row_idx == 0:  # Only print once
            #print("Available columns:", list(data.columns))
            pass
        
        R = float(state['R'])
        Ra = float(state['Ra'])
        K = int(state['K'])
        
        # Debug: Check if we have enough agent columns
        #agent_columns = [col for col in data.columns if col.startswith('agent') and ('_x' in col or '_y' in col)]
        #print(f"Row {row_idx}: K={K}, Found agent columns: {len(agent_columns)//2}")
        
        # Build agent positions more safely
        p_agents = []
        for i in range(1, K+1):
            x_col = f'agent{i}_x'
            y_col = f'agent{i}_y'
            
            if x_col not in data.columns or y_col not in data.columns:
                print(f"Missing columns for agent {i}: {x_col}, {y_col}")
                print(f"Available agent columns: {[col for col in data.columns if 'agent' in col]}")
                raise KeyError(f"Missing agent columns for agent {i}")
            
            p_agents.append([state[x_col], state[y_col]])
        
        p_agents = np.array(p_agents)
        phi_agents = np.array([state[f'agent{i}_phi'] for i in range(1, K+1)])  # antenna directions
        
        p_tx = np.array([state['p_tx_x'], state['p_tx_y']])
        p_rx = np.array([state['p_rx_x'], state['p_rx_y']])
        Rcom = state['Rcom']
        sigma = state['sigma']
        beta = state['beta']

        # Jammer info
        p_jammer = np.array([state['jammer_x'], state['jammer_y']]) 
        v_jammer = np.array([state['jammer_dx'], state['jammer_dy']]) 
        jammer_info = {'p_jammer': p_jammer, 'v_jammer': v_jammer}

        sample = {
            'R': R,
            'Ra': Ra,
            'K': K,
            'p_agents': p_agents,
            'phi_agents': phi_agents,
            'jammer_info': jammer_info,
            'p_tx': p_tx,
            'p_recv': p_rx,  
            'Rcom': Rcom,
            'sigma': sigma,
            'beta': beta
        }

        return sample
        
    except Exception as e:
        print(f"Error in get_state_dict for row {row_idx}: {e}")
        print(f"Row data keys: {list(state.index) if 'state' in locals() else 'state not defined'}")
        raise

def evaluate_baseline(data_file, 
                      c_pos, 
                      c_phi, 
                      rows, 
                      directed_transmission=False, 
                      jammer_on=False, 
                      clustering_on=False, 
                      minimize_distance=False
                      ):
    """
    Evaluate baseline for specified data files and rows.
    
    Args:
        data_files: List of data file names to evaluate
        rows: List of row indices to evaluate for each file
        directed_transmission: Boolean, whether to use directed transmission
        jammer_on: Boolean, whether jammer is active
    
    Returns:
        values: List of evaluation results/values
    """
    results = []
    
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"Warning: File {data_file} not found")
        raise RuntimeError
        
    # Load data file (assuming it's a numpy file or similar)
    try:
        data = pd.read_csv(data_file)
        
        for row_idx in tqdm(rows, desc="Processing rows", unit="row"):
            if row_idx >= len(data):
                print(f"Warning: Row {row_idx} not found in {data_file}")
                continue
            
            # Extract state from row
            #print(f"Evaluating row {row_idx+1}")
            #state = data.iloc[row_idx]
            state = get_state_dict(data, row_idx)
            
            # Parse state data so that Dijkstra can be executed
            K = state['K']
            p_agents = state['p_agents']
            p_tx = state['p_tx']
            p_rx = state['p_recv']
            Rcom = state['Rcom']
            #sigma = state['sigma']
            sigma = 0.2
            beta = state['beta']
            R = state['R']
            #print(row_idx)
            #print(f'R {R}')

            # Directed transmission states
            if directed_transmission:
                phi_agents = state['phi_agents']
            else:
                phi_agents = None

            # Jammer state
            if jammer_on:
                jammer_info = state['jammer_info']
                p_jammer = jammer_info['p_jammer']
                v_jammer = jammer_info['v_jammer']
                jammer_info = {'p_jammer': p_jammer, 'v_jammer': v_jammer}
            else:
                jammer_info = None

            # Run baseline evaluation
            result = baseline(
                p_agents, 
                p_tx,
                p_rx,
                Rcom=Rcom,
                sigma=sigma,
                beta=beta,
                c_pos=c_pos,
                phi_agents=phi_agents,
                jammer_info=jammer_info,
                clustering=clustering_on
            )
            

            results.append({
                'idx': row_idx + 1,  # Use idx as row+1,
                'R': R,
                'value': result['value'],
                'budget': result['budget'],
                'agent_sum_distance': result['agent_sum_distance'],
                #'message_air_distance': result['message_air_distance'],
                'delivery_time': result['delivery_time'],
                'directed_transmission': directed_transmission,
                'jammer_on': jammer_on,
                'clustering_on': clustering_on,
                'minimize_distance': minimize_distance,
                'K': K,
                'file': data_file
            })
                
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def get_rows_by_value(results, threshold, mode='lower'):
    """
    Returns the indices/rows of the states that have a value lower or higher than the specified threshold.
    Args:
        results: List of result dictionaries from evaluate_baseline.
        threshold: Numeric threshold to compare against.
        mode: 'lower' for values < threshold, 'higher' for values > threshold.
    Returns:
        List of row indices.
    """
    if mode == 'lower':
        return [r['idx'] for r in results if 'value' in r and r['value'] < threshold]
    elif mode == 'higher':
        return [r['idx'] for r in results if 'value' in r and r['value'] > threshold]
    else:
        raise ValueError("mode must be 'lower' or 'higher'")
    

def plot_histogram(results, plot_dir=None, c_pos=0, c_phi=0, method=None):
    """
    Plot four histograms in a 2x2 layout for evaluation results:
    1) value, 2) delivery_time, 3) agent_sum_distance, 4) message_air_distance
    """
    metrics = [
        ('value', 'Value', '$V$'),
        ('delivery_time', 'Delivery Time', r'$T_{\mathrm{del}}$'),
        ('agent_sum_distance', 'Agent Sum Distance', r'$D_{\mathrm{tot}}$'),
        ('message_air_distance', 'Message Air Distance', r'$D_{\mathrm{air}}$')
    ]
    K = results[0]['K'] if results else 'N/A'
    directed_transmission = results[0]['directed_transmission'] if results else 'N/A'
    if 'jammer_on' in results[0].keys():
        jammer_on = results[0]['jammer_on'] 
    else:
        jammer_on = False

    if 'clustering_on' in results[0].keys():
        clustering_on = results[0]['clustering_on']
    else:
        clustering_on = False

    plt.figure(figsize=(12, 10))  # Adjusted figure size for 2x2 layout
    
    for i, (key, title, xlabel) in enumerate(metrics):
        values = [r[key] for r in results if key in r]
        if not np.isnan(values[0]):
            plt.subplot(2, 2, i+1)  # Changed to 2x2 layout
            if values:
                plt.hist(values, bins=35, alpha=0.7, edgecolor='black', density=True)
                if key == 'value':
                    plt.axvline(0.0, color='red', linestyle='--', linewidth=2)
            plt.xlabel(xlabel)
            plt.ylabel('density')
            plt.title(title)
            plt.grid(True, alpha=0.3)

    if method is None:
        method = "Baseline"
    sup_title = f'{method} evaluations K={K}, directed={directed_transmission}, jammer={jammer_on}, clustering={clustering_on}'
    sup_title += fr'\n $c_{{\delta p}}=${c_pos}, $c_{{\delta\phi}}=${c_phi}'
    plt.suptitle(sup_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjusted rect to accommodate suptitle

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'evaluation_histograms_K{K}_dir{int(bool(directed_transmission))}_jam{int(bool(jammer_on))}_clust{int(bool(clustering_on))}.pdf')
    
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved histogram plot to {plot_path}")

    #plt.show()

def get_config_string(directed_transmission, jammer_on, c_pos, c_phi):
    config_string = f"dir{int(bool(directed_transmission))}_jam{int(bool(jammer_on))}_cpos{c_pos}_cphi{c_phi}"  

    return config_string

def get_marl_config_string(method, K, c_pos, c_phi, directed_transmission, jammer_on, testing=False):
    if testing:
        test_string = "testing"
    else:
        test_string = "evaluation"
    config_string = f"{method}_{test_string}_results_K{K}_cpos{c_pos}_cphi{c_phi}_n10000_dir{int(bool(directed_transmission))}_jam{int(bool(jammer_on))}"  

    return config_string

def save_evaluation_results(results, eval_result_dir, K, c_pos, c_phi, n, 
                            directed_transmission=False, jammer_on=False, clustering_on=False, minimize_distance=False):
    """
    Save evaluation results to a CSV file in a structured directory.
    """
    if not results:
        print("No results to save.")
        return

    df_results = pd.DataFrame(results)
    # Compose directory and file name
    conf_string = get_config_string(directed_transmission, jammer_on, c_pos=c_pos, c_phi=c_phi)
    eval_result_dir = os.path.join(eval_result_dir, conf_string)
    os.makedirs(eval_result_dir, exist_ok=True)
    eval_file = f"evaluation_results_K{K}_n{n}_{conf_string}.csv"
    eval_path = os.path.join(eval_result_dir, eval_file)
    
    # Save to CSV
    df_results.to_csv(eval_path, index=False)
    print(f"Saved evaluation results to {eval_path}")

# Add this function after the plot_histogram function:

def plot_comparison_histograms(differences, method1, method2, plot_dir=None, k=None):
    """
    Plot four histograms in a 2x2 layout showing differences between two configurations:
    1) value_diff, 2) delivery_time_diff, 3) agent_sum_distance_diff, 4) message_air_distance_diff
    """
    metrics = [
        ('value_diff', 'Value Difference', r'$\Delta V$'),
        #('delivery_time_diff', 'Delivery Time Difference', fr'$T_{{\mathrm{{del, {method2}}}}}/T_{{\mathrm{{del, {method1}}}}}$ %'),
        ('delivery_time_diff', 'Delivery Time Ratio', fr'$T_{{\mathrm{{del}}}}$ %'),
        ('agent_sum_distance_diff', 'Agent Sum Distance Difference', r'$\Delta D_{\mathrm{tot}}$'),
        ('message_air_distance_diff', 'Message Air Distance Ratio', fr'$D_{{\mathrm{{air}}}}$ %')
    ]
    
    plt.figure(figsize=(12, 10))
    
    for i, (key, title, xlabel) in enumerate(metrics):
        values = [d[key] for d in differences if key in d]
        plt.subplot(2, 2, i+1)
        
        if values:
            plt.hist(values, bins=50, alpha=0.7, edgecolor='black', density=True)
            # Add vertical line at zero for reference
            if i in [0,2]:
                plt.axvline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.8)
            else:
                plt.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            plt.text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.xlabel(xlabel)
        plt.ylabel('density')
        plt.title(title)
        plt.grid(True, alpha=0.3)

    sup_title = f'Policy Comparison: ({method2}) vs ({method1})\nK={k}'
    plt.suptitle(sup_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'comparison_histograms_K{k}_{method1}_vs_{method2}.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved comparison histogram to {plot_path}")
    else:
        plt.show()


def plot_comparison_scatter(results1, results2, method1, method2, plot_dir=None, k=None):
    """
    Plot four scatter plots in a 2x2 layout comparing two configurations:
    1) value, 2) delivery_time, 3) agent_sum_distance, 4) message_air_distance
    X-axis: method1 values, Y-axis: method2 values
    """
    metrics = [
        ('value', 'Value', '$V$'),
        ('delivery_time', 'Delivery Time', r'$T_{{\mathrm{del}}}$'),
        ('agent_sum_distance', 'Agent Sum Distance', r'$D_{{\mathrm{tot}}}$'),
        ('message_air_distance', 'Message Air Distance', r'$D_{{\mathrm{air}}}$')
    ]
    
    plt.figure(figsize=(12, 10))
    
    for i, (key, title, xlabel) in enumerate(metrics):
        values1 = [r[key] for r in results1 if key in r]
        values2 = [r[key] for r in results2 if key in r]
        
        plt.subplot(2, 2, i+1)
        
        if values1 and values2:
            # Create scatter plot
            plt.scatter(values1, values2, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Get axis limits
            min_val = min(min(values1), min(values2))
            max_val = max(max(values1), max(values2))
            
            # Add padding
            padding = (max_val - min_val) * 0.05
            min_val -= padding
            max_val += padding
            
            # Set equal axis limits
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            plt.gca().set_aspect('equal')
            
            # Plot diagonal reference line (y=x)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Equal performance')
            
            # Add statistics text
            correlation = np.corrcoef(values1, values2)[0, 1]
            plt.text(0.05, 0.95, f'ρ={correlation:.3f}\nn={len(values1)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.xlabel(f'{xlabel} ({method1})', fontsize=10)
        plt.ylabel(f'{xlabel} ({method2})', fontsize=10)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', fontsize=9)

    sup_title = f'Policy Comparison: ({method2}) vs ({method1})\nK={k}'
    plt.suptitle(sup_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'comparison_scatter_K{k}_{method1}_vs_{method2}.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved comparison scatter plot to {plot_path}")
    else:
        plt.show()


def plot_comparison_heatmap(results1, results2, method1, method2, plot_dir=None, k=None, detailed=False):
    """
    Plot comparison heatmaps using seaborn for three metrics.
    Each heatmap shows the 2D density of (method1, method2) pairs for a given metric.
    X-axis: method1 values, Y-axis: method2 values
    
    Args:
        results1: Results list for method1
        results2: Results list for method2
        method1: Name of first method
        method2: Name of second method
        plot_dir: Directory to save plots
        k: K value for the title
        detailed: Boolean, if True adds statistics text boxes to each heatmap
    """

    # Set Computer Modern font for LaTeX compatibility
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True

    metrics = [
        ('value', 'Value', '$V$'),
        ('delivery_time', 'Delivery Time', r'$T_{{\mathrm{del}}}$'),
        ('agent_sum_distance', 'Agent Sum Distance', r'$D_{{\mathrm{tot}}}$')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, (key, title, xlabel) in enumerate(metrics):
        values1 = np.array([r[key] for r in results1 if key in r])
        values2 = np.array([r[key] for r in results2 if key in r])
        
        ax = axes[i]
        
        if len(values1) > 0 and len(values2) > 0:
            # Get axis limits and use equal range for both x and y
            min_val1 = np.min(values1)
            max_val1 = np.max(values1)
            min_val2 = np.min(values2)
            max_val2 = np.max(values2)
            
            # Use equal axis limits for both x and y
            min_val = min(min_val1, min_val2)
            max_val = max(max_val1, max_val2)

            # For value metric, ensure minimum is at most 0
            if key == 'value':
                min_val = min(0, min_val)
            
            # Add padding
            padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
            
            min_val = min_val - padding
            max_val = max_val + padding
            
            # Create DataFrame for seaborn
            df_heatmap = pd.DataFrame({
                f'{method1}': values1,
                f'{method2}': values2
            })
            
            # Create 2D histogram bins
            n_bins = 50
            bins = np.linspace(min_val, max_val, n_bins)
            
            # Digitize values into bins
            x_binned = np.digitize(values1, bins) - 1
            y_binned = np.digitize(values2, bins) - 1
            
            # Create heatmap matrix
            heatmap_matrix = np.zeros((len(bins)-1, len(bins)-1))
            for xi, yi in zip(x_binned, y_binned):
                if 0 <= xi < len(bins)-1 and 0 <= yi < len(bins)-1:
                    heatmap_matrix[yi, xi] += 1
            
            # Plot heatmap using seaborn without colorbar or gridlines
            hm = sns.heatmap(
                heatmap_matrix,
                ax=ax,
                cmap='viridis',
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                square=True,
                linewidths=0,
                linecolor=None
            )

            # --- Force-disable all edge and antialias artifacts ---
            for coll in ax.collections:
                coll.set_edgecolor("none")
                coll.set_linewidth(0)
                coll.set_antialiased(False)
                coll.set_rasterized(True)  # rasterize to eliminate vector edge seams

            # Remove any matplotlib grid and spines
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Invert y-axis so it goes from bottom to top
            ax.invert_yaxis()
            
            # Plot diagonal reference line (y=x)
            ax.plot([0, len(bins)-1], [0, len(bins)-1], 'r--', alpha=0.5, linewidth=1.5, label='Equal performance')
            
            # Add vertical and horizontal lines at 0 for value heatmap
            if key == 'value':
                zero_pos = (0 - min_val) / (max_val - min_val) * (len(bins) - 1)
                ax.axvline(x=zero_pos, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
                ax.axhline(y=zero_pos, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

            # Set equal aspect
            ax.set_aspect('equal', adjustable='box')
            
            # Create tick labels
            tick_positions = np.linspace(0, len(bins)-1, 5)
            tick_labels = [f'{min_val + (max_val - min_val) * i / (len(bins)-1):.2f}' 
                          for i in range(0, len(bins), len(bins)//4)]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            
            # Add 0 tick marks on axes for value heatmap (leftmost)
            if key == 'value':
                # Find position of 0 in the bins
                zero_pos = (0 - min_val) / (max_val - min_val) * (len(bins) - 1)
                
                # Set only three ticks: min, zero, max
                min_pos = 0
                max_pos = len(bins) - 1
                
                tick_positions = [min_pos, zero_pos, max_pos]
                tick_labels = [f'{min_val:.2f}', f'0.00', f'{max_val:.2f}']
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
            
            # Add statistics text box if detailed is True
            if detailed:
                median_diff = np.median(values2 - values1)
                std_diff = np.std(values2 - values1)
                median_val1 = np.median(values1)
                median_val2 = np.median(values2)
                
                stats_text = (f'$\\mathrm{{med}}_{{{method1}}}$={median_val1:.3f}\n'
                             f'$\\mathrm{{med}}_{{{method2}}}$={median_val2:.3f}\n'
                             f'$\\Delta\\mathrm{{med}}$={median_diff:.3f}\n'
                             f'$\\sigma(\\Delta)$={std_diff:.3f}')
                ax.text(0.05, 0.9, stats_text, transform=ax.transAxes, fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor='gray'))
        
        ax.set_xlabel(f'{xlabel} ({method1})', fontsize=10)
        ax.set_ylabel(f'{xlabel} ({method2})', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper left', fontsize=9)

    sup_title = f'Policy Comparison Heatmap: ({method2}) vs ({method1}) for K={k}'
    #fig.suptitle(sup_title, fontsize=14)
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.12)


    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'comparison_heatmap_K{k}_{method1}_vs_{method2}.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved comparison heatmap to {plot_path}")
    else:
        plt.show()


def compute_tx_communication_radius(p_rec, p_jammer):
    Rcom = np.sqrt(1 / (1 + 3.0 * np.linalg.norm(p_rec - p_jammer)**(-2)))

    return Rcom


def save_trajectory(baseline_result, savepath, file=None, directed_transmission=False, jammer_on=False, file_idx=0):
    R = baseline_result['R']
    p_trajectories = baseline_result['p_trajectories']
    phi_trajectories = baseline_result['phi_trajectories']
    has_message_trajectories = baseline_result['has_message']
    p_jammer_trajectory = baseline_result['p_jammer_trajectory']
    #active_agents = baseline_result['active_agents']

    # Extract the finish time
    t_finish = -1
    for k in p_trajectories.keys():
        traj_k = p_trajectories[k]
        
        t_k = max(traj_k.keys()) 
        if t_k > t_finish:
            t_finish = t_k


    K = len(p_trajectories)
    trajectories = {}
    for idx, k in enumerate(p_trajectories.keys()):
        p_trajectories_k = p_trajectories[k]
        phi_trajectories_k = phi_trajectories[k]
        has_message_trajectories_k = has_message_trajectories[k]
        t_start_k = min(p_trajectories_k.keys())
        t_stop_k = max(p_trajectories_k.keys())

        # Create array with full length trajectory, potentially padding with initial and final positions
        p_traj_full_k = []
        phi_full_k = []
        has_message_full_k = []
        for t in range(t_finish+1):
            if t < t_start_k:
                p_traj_full_k.append(p_trajectories_k[t_start_k])
                phi_full_k.append(phi_trajectories_k[t_start_k])
                has_message_full_k.append(False)
            elif t > t_stop_k:
                p_traj_full_k.append(p_trajectories_k[t_stop_k])
                phi_full_k.append(phi_trajectories_k[t_stop_k])
                has_message_full_k.append(has_message_trajectories_k[t_stop_k])
            else:
                p_traj_full_k.append(p_trajectories_k[t])
                phi_full_k.append(phi_trajectories_k[t])
                has_message_full_k.append(has_message_trajectories_k[t])

        traj_k = {'p_traj': p_traj_full_k, 'phi_traj': phi_full_k, 'has_message': has_message_full_k}
        trajectories[idx] = traj_k

    # Jammer trajectory
    if jammer_on:
        p_jammer_traj_full = []
        for t in range(t_finish+1): 
            p_jammer_traj_full.append(p_jammer_trajectory[t])
        

    records = []

    for t in range(t_finish+1):
        # Save as a flat record
        record = {
            'idx': t+1,  
            't': t,
            'R': R,
            'directed_transmission': directed_transmission,
            'jammer_on': jammer_on
        }
        # Add agent positions and orientations
        for k in range(K):
            traj_k = trajectories[k]
            p_agents = traj_k['p_traj']
            phi_agents = traj_k['phi_traj']
            has_message = traj_k['has_message']

            record[f'agent{k+1}_x'] = p_agents[t][0]
            record[f'agent{k+1}_y'] = p_agents[t][1]
            record[f'agent{k+1}_phi'] = phi_agents[t]
            record[f'agent{k+1}_has_message'] = has_message[t]
            
            #record[f'agent{k+1}_active'] = True if (k) in active_agents else False
        records.append(record)  

        # Jammer position
        if jammer_on:
            if p_jammer_traj_full[0] is not None:
                record[f'jammer_x'] = p_jammer_traj_full[t][0]
                record[f'jammer_y'] = p_jammer_traj_full[t][1]
            else:
                record[f'jammer_x'] = None
                record[f'jammer_y'] = None

        if file:
            record['file_idx'] = file_idx
            record['file'] = file


    df = pd.DataFrame(records)
    df.to_csv(savepath, index=False)
    print(f"Saved trajectory of length {t_finish} to {savepath}")


def plot_comparison_heatmap_matrix(results_dict, methods, plot_dir=None, k=None):
    """
    Plot an NxN matrix of comparison heatmaps between all pairs of methods.
    Diagonal entries are left empty.
    
    Args:
        results_dict: Dictionary with method names as keys and results lists as values
        methods: List of method names in the order they should appear
        plot_dir: Directory to save plots
        k: K value for the title
    """
    n_methods = len(methods)
    metrics = ['value', 'delivery_time', 'agent_sum_distance', 'message_air_distance']
    
    # Create figure with NxN subplots for each metric
    fig, axes = plt.subplots(n_methods, n_methods, figsize=(4*n_methods, 4*n_methods))
    
    if n_methods == 1:
        axes = np.array([[axes]])
    elif n_methods == 2:
        axes = axes.reshape(n_methods, n_methods)
    
    for i, method_y in enumerate(methods):
        for j, method_x in enumerate(methods):
            ax = axes[i, j]
            
            # Diagonal: leave empty but keep axis visible
            if i == j:
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.text(0.5, 0.5, method_x, ha='center', va='center', 
                       fontsize=14, fontweight='bold', transform=ax.transAxes)
                continue
            
            # Get results
            results_x = results_dict[method_x]
            results_y = results_dict[method_y]
            
            # Trim to minimum length
            min_len = min(len(results_x), len(results_y))
            results_x = results_x[:min_len]
            results_y = results_y[:min_len]
            
            # Use first metric for the matrix heatmap (value)
            key = 'value'
            values_x = np.array([r[key] for r in results_x if key in r])
            values_y = np.array([r[key] for r in results_y if key in r])
            
            if len(values_x) > 0 and len(values_y) > 0:
                # Get axis limits and use equal range
                min_val_x = np.min(values_x)
                max_val_x = np.max(values_x)
                min_val_y = np.min(values_y)
                max_val_y = np.max(values_y)
                
                min_val = min(min_val_x, min_val_y)
                max_val = max(max_val_x, max_val_y)
                
                # Add padding
                padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
                min_val = min_val - padding
                max_val = max_val + padding
                
                # Create 2D histogram
                bins = [np.linspace(min_val, max_val, 20), np.linspace(min_val, max_val, 20)]
                heatmap, xedges, yedges = np.histogram2d(values_x, values_y, bins=bins)
                heatmap = heatmap.T
                
                # Plot heatmap
                im = ax.imshow(heatmap, origin='lower', aspect='auto', interpolation='nearest',
                              extent=[min_val, max_val, min_val, max_val],
                              cmap='viridis', alpha=0.8)
                
                # Plot diagonal reference line
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
                
                # Set axis limits and aspect
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
                ax.set_aspect('equal', adjustable='box')
                
                # Set equal number of ticks on both axes - 5 ticks
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=False))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=False))
                
                # Format tick labels to show numbers
                ax.tick_params(axis='both', labelsize=9)
                
                # Add statistics
                correlation = np.corrcoef(values_x, values_y)[0, 1]
                mean_diff = np.mean(values_y - values_x)
                std_diff = np.std(values_y - values_x)
                
                stats_text = f'ρ={correlation:.2f}\nμ={mean_diff:.2f}\nσ={std_diff:.2f}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # Set labels for outer axes only
            if i == n_methods - 1:
                ax.set_xlabel(method_x, fontsize=11, fontweight='bold')
                ax.xaxis.tick_bottom()
            else:
                ax.tick_params(labelbottom=False)
            
            if j == 0:
                ax.set_ylabel(method_y, fontsize=11, fontweight='bold')
                ax.yaxis.tick_left()
            else:
                ax.tick_params(labelleft=False)
            
            # Grid
            ax.grid(True, alpha=0.2, linestyle=':')
    
        # Add overall title
    fig.suptitle('Pairwise Comparison Heatmap Matrix (Value)',
             fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, f'K={k}', fontsize=13, ha='center')

    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'comparison_heatmap_matrix_K{k}.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved comparison heatmap matrix to {plot_path}")
    else:
        plt.show()


def plot_all_pairwise_comparisons(results_dict, methods, compare_mode="heatmap", plot_dir=None, k=None):
    """
    Plot all pairwise comparisons between multiple methods.
    
    Args:
        results_dict: Dictionary with method names as keys and results lists as values
        methods: List of method names in the order they should be compared
        compare_mode: "hist", "scatter", or "heatmap"
        plot_dir: Directory to save plots
        k: K value for the title
    """
    from itertools import combinations
    
    # Generate all pairwise combinations
    method_pairs = list(combinations(methods, 2))
    
    metrics = ['value', 'delivery_time', 'agent_sum_distance', 'message_air_distance']
    
    for method1, method2 in method_pairs:
        results1 = results_dict[method1]
        results2 = results_dict[method2]
        
        # Trim to minimum length
        min_len = min(len(results1), len(results2))
        results1 = results1[:min_len]
        results2 = results2[:min_len]
        
        print(f"\n{'='*60}")
        print(f"Comparison: {method2} vs {method1}")
        print(f"{'='*60}")
        
        # Compute differences
        differences = []
        for j in range(min_len):
            diff_dict = {
                'idx': results1[j]['idx'],
                'K': k,
                'value_diff': results2[j]['value'] - results1[j]['value'],
                'delivery_time_diff': results2[j]['delivery_time'] / results1[j]['delivery_time'],
                'agent_sum_distance_diff': results2[j]['agent_sum_distance'] - results1[j]['agent_sum_distance'],
                'message_air_distance_diff': results2[j]['message_air_distance'] / results1[j]['message_air_distance']
            }
            differences.append(diff_dict)
        
        # Plot based on mode
        if compare_mode == "hist":
            plot_comparison_histograms(differences, method1, method2, plot_dir=plot_dir, k=k)
        elif compare_mode == "scatter":
            plot_comparison_scatter(results1, results2, method1, method2, plot_dir=plot_dir, k=k)
        elif compare_mode == "heatmap":
            plot_comparison_heatmap(results1, results2, method1, method2, plot_dir=plot_dir, k=k)
        
        # Print summary statistics
        print(f"\nComparison Summary ({method2} - {method1}):")
        for metric in metrics:
            diff_values = [d[f'{metric}_diff'] for d in differences]
            mean_diff = np.mean(diff_values)
            std_diff = np.std(diff_values)
            print(f"  {metric}: mean={mean_diff:.4f}, std={std_diff:.4f}")
            

def plot_violin_plots(results_input, plot_dir=None, c_pos=0, c_phi=0, directed_transmission=False, jammer_on=False):
    """
    Plot violin plots for three metrics (value, delivery_time, agent_sum_distance) across different K values.
    Can handle either a single results dict (K values as keys) or multiple results dicts (method names as keys).
    
    Args:
        results_input: Either:
            - Dictionary with K values as keys and results lists as values (single method)
            - Dictionary with method names as keys and K-indexed result dicts as values (multiple methods)
        plot_dir: Directory to save plots
        c_pos: Position cost parameter
        c_phi: Antenna cost parameter
    """
    metrics = [
        ('value', 'Value', '$V$'),
        ('delivery_time', 'Delivery Time', fr'$T_{{\mathrm{{del}}}}$'),
        ('agent_sum_distance', 'Agent Sum Distance', fr'$D_{{\mathrm{{tot}}}}$')
    ]
    
    # Determine if input is single dict or dict of dicts
    is_multi_method = False
    if results_input:
        first_key = list(results_input.keys())[0]
        first_value = results_input[first_key]
        # If the value is a dict (of K values), it's multi-method
        if isinstance(first_value, dict):
            is_multi_method = True
    
    # Process single method case
    if not is_multi_method:
        results_dict = results_input
        K_values = sorted(results_dict.keys())
        methods = [None]  # Single method, no label needed
        results_dicts_list = [results_dict]
    else:
        # Process multiple methods case
        methods = sorted(results_input.keys())
        K_values = sorted(results_input[methods[0]].keys())
        results_dicts_list = [results_input[method] for method in methods]
    
    # Create figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    for idx, (key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data in long format for seaborn
        data_list = []
        for method_idx, method in enumerate(methods):
            results_dict = results_dicts_list[method_idx]
            for k in K_values:
                results = results_dict[k]
                for r in results:
                    if key in r and not np.isnan(r[key]):
                        if is_multi_method:
                            # For multiple methods, group by method and K
                            data_list.append({
                                'K': str(k),
                                'Method': method,
                                'Value': r[key]
                            })
                        else:
                            # For single method, just group by K
                            data_list.append({
                                'K': str(k),
                                'Value': r[key]
                            })
        
        if data_list:
            df_plot = pd.DataFrame(data_list)
            
            # Create violin plot using seaborn
            if is_multi_method:
                # Multiple methods: use hue to separate by method
                sns.violinplot(data=df_plot, x='K', y='Value', hue='Method', ax=ax, 
                              inner='box', linewidth=2)
                ax.legend(title='Method', loc='upper left', fontsize=10)
            else:
                # Single method: uniform blue color
                sns.violinplot(data=df_plot, x='K', y='Value', ax=ax, 
                              inner='box', color='skyblue', linewidth=2)

            # Add horizontal reference line at y=0 for 'value' metric
            if key == 'value':
                ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Reference (value=0)')
            
            # Customize plot
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_xlabel('K', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add mean and std text for each K (only for single method to avoid clutter)
            if not is_multi_method:
                # Add single label in top left corner instead of per-K labels
                if idx == 0:  # Only add once per subplot
                    results = results_dicts_list[0][K_values[0]]
                    values = [r[key] for r in results if key in r and not np.isnan(r[key])]
                    if values:
                        # Collect stats for all K values
                        all_stats = []
                        for k in K_values:
                            results_k = results_dicts_list[0][k]
                            values_k = [r[key] for r in results_k if key in r and not np.isnan(r[key])]
                            if values_k:
                                all_stats.append(f'K={k}: μ={np.mean(values_k):.2f}')
                        
                        stats_text = '\n'.join(all_stats)
                        ax.text(0.02, 0.98, stats_text, 
                               transform=ax.transAxes, fontsize=8, va='top', ha='left',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for {title}', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    # Overall title
    if is_multi_method:
        method_str = " vs ".join(methods)
        sup_title = fr'{method_str}\n$c_{{\delta p}}=${c_pos}, $c_{{\delta\phi}}=${c_phi}\ndirected_transmission={directed_transmission}, jammer_on={jammer_on}'
    else:
        sup_title = fr'Metrics vs K\n$c_{{\delta p}}=${c_pos}, $c_{{\delta\phi}}=${c_phi}\ndirected_transmission={directed_transmission}, jammer_on={jammer_on}'
    
    fig.suptitle(sup_title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        if is_multi_method:
            method_str = "_vs_".join(methods)
            plot_path = os.path.join(plot_dir, f'violin_plots_compare_{method_str}_cpos{c_pos}_cphi{c_phi}.pdf')
        else:
            plot_path = os.path.join(plot_dir, f'violin_plots_cpos{c_pos}_cphi{c_phi}.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved violin plots to {plot_path}")
    else:
        plt.show()


def plot_comparison_heatmap_all(comparisons, eval_K, plot_dir=None, methods_str="", scenario_str=""):
    """
    Plot a grid of comparison heatmaps.
    Rows: Different method comparisons
    Columns: metrics (value, delivery_time, agent_sum_distance)
    All comparisons use the same axis limits per metric.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Set Computer Modern font for LaTeX compatibility
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    # Colormaps for the Ks
    colormaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'YlOrBr']

    # Ensure all colormaps show masked entries as white
    for cm in colormaps:
        plt.get_cmap(cm).set_bad('white')

    metrics = [
        ('value', 'Value', '$V$'),
        ('delivery_time', 'Delivery Time', r'$T_{\mathrm{del}}$'),
        ('agent_sum_distance', 'Agent Sum Distance', r'$D_{\mathrm{tot}}$')
    ]

    fig, axes = plt.subplots(len(comparisons), 3, figsize=(16, 5*len(comparisons)))

    if len(comparisons) == 1:
        axes = axes.reshape(1, -1)

    # threshold for white-out of sparse bins (normalized scale)
    threshold = 0.05

    # ------------------------------------------
    # Compute GLOBAL axis limits across ALL comparisons and K values
    # ------------------------------------------
    global_limits = {}
    
    for col_idx, (key, title, xlabel) in enumerate(metrics):
        all_vals1_global = []
        all_vals2_global = []
        
        for method1, method2, results1_dict, results2_dict in comparisons:
            for k in eval_K:
                if k in results1_dict and k in results2_dict:
                    r1 = results1_dict[k]
                    r2 = results2_dict[k]
                    
                    v1 = np.array([x[key] for x in r1 if key in x])
                    v2 = np.array([x[key] for x in r2 if key in x])
                    all_vals1_global.extend(v1)
                    all_vals2_global.extend(v2)
        
        if all_vals1_global and all_vals2_global:
            min_val = min(np.min(all_vals1_global), np.min(all_vals2_global))
            max_val = max(np.max(all_vals1_global), np.max(all_vals2_global))
            
            # ensure value metric includes 0
            if key == 'value':
                min_val = min(min_val, 0)
            
            # padding
            #pad = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
            #min_val -= pad
            #max_val += pad
            
            global_limits[key] = (min_val, max_val)
        else:
            global_limits[key] = (0, 1)

    # ------------------------------------------
    # Begin main loops
    # ------------------------------------------

    for row_idx, (method1, method2, results1_dict, results2_dict) in enumerate(comparisons):
        print(f"\nCreating comparison heatmaps: {method1} vs {method2} across all K values...")

        for col_idx, (key, title, xlabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Use global limits
            min_val, max_val = global_limits[key]

            # consistent bins
            n_bins = 50
            bins = np.linspace(min_val, max_val, n_bins)

            # ------------------------------------------
            # Compute global max count across all Ks (per comparison)
            # ------------------------------------------
            heatmaps_by_k = {}
            global_max = 0

            for k in eval_K:
                if k in results1_dict and k in results2_dict:
                    r1 = results1_dict[k]
                    r2 = results2_dict[k]
                    m = min(len(r1), len(r2))
                    r1, r2 = r1[:m], r2[:m]

                    v1 = np.array([x[key] for x in r1 if key in x])
                    v2 = np.array([x[key] for x in r2 if key in x])

                    xb = np.digitize(v1, bins) - 1
                    yb = np.digitize(v2, bins) - 1

                    H = np.zeros((n_bins-1, n_bins-1))
                    for xi, yi in zip(xb, yb):
                        if 0 <= xi < n_bins-1 and 0 <= yi < n_bins-1:
                            H[yi, xi] += 1

                    heatmaps_by_k[k] = H
                    global_max = max(global_max, H.max())

            global_max = max(global_max, 1)  # avoid divide-by-zero

            # ------------------------------------------
            # Plot each K's normalized heatmap
            # ------------------------------------------
            for k_idx, k in enumerate(eval_K):
                if k not in heatmaps_by_k:
                    continue

                H = heatmaps_by_k[k]
                H_norm = H / global_max

                # mask low intensities → white
                H_mask = np.ma.masked_less(H_norm, threshold)

                cmap = plt.get_cmap(colormaps[k_idx % len(colormaps)])

                alpha_val = 0.8  # Increase from 0.6 to make colors stronger in PDF
                im = ax.imshow(H_mask, origin='lower', cmap=cmap,
                            extent=[min_val, max_val, min_val, max_val],
                            alpha=alpha_val, interpolation='nearest')

                print(f"  K={k}: plotted with {H.sum()} total counts")

            # ------------------------------------------
            # Plot decorations
            # ------------------------------------------
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', alpha=0.4, linewidth=1.2)

            #if key == 'value':
            #    ax.axvline(0, color='black', linewidth=1, alpha=0.4)
            #    ax.axhline(0, color='black', linewidth=1, alpha=0.4)

            ax.set_aspect('equal')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            # ticks
            fontsize = 20
            ticks = np.linspace(min_val, max_val, 5)
            labels = [f"{t:.0f}" for t in ticks]

            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=0, ha='right', fontsize=fontsize)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=fontsize)

            # legend for K
            handles = [
                plt.Rectangle((0, 0), 1, 1,
                               color=plt.get_cmap(colormaps[i % len(colormaps)])(0.7),
                               label=f'$K$={k}')
                for i, k in enumerate(eval_K)
            ]
            ax.legend(handles=handles, fontsize=fontsize-5, loc='upper left')

            # Only set title for top row
            if row_idx == 0:
                ax.set_title(r'\textbf{' + title + '}', fontsize=fontsize)
            
            # Move labels inside the frame (bottom-left corner positioning)
                        # Move labels to middle of corresponding axes
            ax.text(0.5, 0.02, f'{method1}', fontsize=fontsize, 
                   transform=ax.transAxes, va='bottom', ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            ax.text(0.02, 0.5, f'{method2}', fontsize=fontsize, 
                   transform=ax.transAxes, va='center', ha='left', rotation=90,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))


    plt.subplots_adjust(wspace=0.1, hspace=0.25, left=0.08, right=0.8, top=0.88, bottom=0.12)


    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        # Create descriptive filename
        if methods_str and scenario_str:
            plot_filename = f'comparison_heatmaps_{methods_str}_{scenario_str}.pdf'
        else:
            plot_filename = f'comparison_heatmaps_allK.pdf'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
        print(f"\nSaved figure to: {plot_path}")
    else:
        plt.show()

def generate_baseline_trajectory(k, row, data_dir, c_pos, c_phi, directed_transmission=False, jammer_on=False, testing=False, no_metrics=False):
    row_idx = row-1
    
    if testing:
        data_name = f"test_states_K{k}_n10000.csv"
    else:
        data_name = f"evaluation_states_K{k}_n10000.csv"
    data_file_path = data_dir + "/" + data_name
    data = pd.read_csv(data_file_path)
    print(f'Loaded {data_file_path}')
    state = get_state_dict(data, row_idx)

    baseline_result = baseline_result(
                state['p_agents'], 
                state['p_tx'],
                state['p_recv'],
                Rcom=state['Rcom'],
                sigma=0.2,
                beta=state['beta'],
                c_pos=c_pos,
                c_phi=c_phi,
                phi_agents=state['phi_agents'] if directed_transmission else None,
                jammer_info=state['jammer_info'] if jammer_on else None,
                clustering=True,
                no_metrics=no_metrics
            )
    
    # Convert trajectory dict to csv and save
    if testing:
        traj_path = "Testing/Data/Trajectories/Baseline"
    else:
        traj_path = "Evaluation/Trajectories/Baseline"
    savepath = f"{traj_path}/baseline_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    save_trajectory(baseline_result, savepath=savepath, file=data_name, directed_transmission=directed_transmission, jammer_on=jammer_on, file_idx=row_idx)


def plot_antenna_lobe(ax, position, phi, Rcom, jammer_pos=None, color='orange'):
    """
    Plot antenna lobe (directional communication range) at a given position.
    Uses the communication_range function to compute the actual directional pattern.
    
    Args:
        ax: matplotlib axis
        position: np.array([x, y]) position of the antenna
        phi: antenna direction angle (in radians)
        Rcom: communication range (used as reference)
        color: color of the lobe (default: 'orange')
    """
    # Create theta range for directed transmission
    theta_eps = 0.05
    theta_range_dir = np.linspace(-np.pi/2 + theta_eps, np.pi/2 - theta_eps, 500)
    
    # Calculate communication range for this phi (directed, no jammer)
    SINR_threshold = 1.0  # minimum SINR for communication
    ranges_dir = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, 
                                      jammer_pos=jammer_pos, p_tx=position) for theta in theta_range_dir]
    
    # Convert theta to actual angles (relative to antenna orientation phi)
    actual_angles_dir = theta_range_dir + phi
    
    # Convert to Cartesian coordinates
    range_x_dir = position[0] + np.array(ranges_dir) * np.cos(actual_angles_dir)
    range_y_dir = position[1] + np.array(ranges_dir) * np.sin(actual_angles_dir)
    
    # Plot the communication range pattern
    fill_alpha = 0.2
    line_alpha = 0.7
    linewidth = 2
    ax.fill(range_x_dir, range_y_dir, alpha=fill_alpha, color=color, zorder=2)
    ax.plot(range_x_dir, range_y_dir, color=color, linewidth=linewidth, 
           alpha=line_alpha, zorder=2)
    
    # Draw antenna boresight arrows
    max_range = max(ranges_dir) if ranges_dir else Rcom
    antenna_length = 0.8 * max_range
    antenna_end = position + antenna_length * np.array([np.cos(phi), np.sin(phi)])
    
    ax.arrow(position[0], position[1], 
            antenna_end[0] - position[0], antenna_end[1] - position[1], 
            head_width=0.04, head_length=0.04, fc=color, ec=color, 
            linewidth=linewidth, alpha=line_alpha, zorder=3)


def plot_trajectory(traj_dir, k, row, directed_transmission, jammer_on, method="Baseline", plot_dir=None):
    """
    Load and plot trajectory from a CSV file in a single figure.
    Agents that don't move are plotted as passive agents in black.
    
    Args:
        traj_dir: Directory containing trajectory files
        k: Number of agents
        row: Row index
        directed_transmission: Boolean, whether directed transmission was used
        jammer_on: Boolean, whether jammer was active
        method: Method name (Baseline, MADDPG, MAPPO)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Set up LaTeX rendering and seaborn style
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 14
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    Rcom = 1.0
    agent_color = 'orange'  # Color for moving agents
    passive_color = 'black'  # Color for stationary agents
    
    # Construct trajectory file name
    traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
    traj_file_path = os.path.join(traj_dir, traj_filename)
    
    if not os.path.exists(traj_file_path):
        print(f"Trajectory file not found: {traj_file_path}")
        return
    
    # Load trajectory data
    traj_data = pd.read_csv(traj_file_path)
    print(f"Loaded trajectory from {traj_file_path}")
    print(f"Trajectory length: {len(traj_data)} time steps")

    # Get list of active agents
    active_agents = []
    for agent_id in range(1, k+1):
        agent_col_has_message = f'agent{agent_id}_has_message'
        if np.any(traj_data[agent_col_has_message]):
            active_agents.append(agent_id)
                
    # Extract R value from data
    R = traj_data['R'].iloc[0] if 'R' in traj_data.columns else 10.0
    Ra = 0.6 * R
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Rcom circle around transmitter
    if jammer_on:
        # Compute communication radius at origin with jammer present

        # Time point at which retrieving agent has message
        t_retrieve = len(traj_data) - 1
        for agent_k in range(1, k+1):
            agent_col_has_message = f'agent{agent_k}_has_message'
            times_with_message = traj_data.index[traj_data[agent_col_has_message] == True].tolist()
            if times_with_message:
                t_retrieve = min(t_retrieve, times_with_message[0])
        t_retrieve -= 1

        # Compute jammer position at t_closest
        jammer_x = traj_data['jammer_x'].iloc[t_retrieve] if 'jammer_x' in traj_data.columns else R/2
        jammer_y = traj_data['jammer_y'].iloc[t_retrieve] if 'jammer_y' in traj_data.columns else 0.0
        p_jammer = np.array([jammer_x, jammer_y])

        Rcom_tx = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=np.array([0, 0]))

        # Compute time point and position where any agent is closest to the origin
        """
        min_distance_to_origin = float('inf')
        t_closest = 0
        closest_agent_id = 1

        for t in range(len(traj_data)):
            for agent_id in range(1, k+1):
                agent_col_x = f'agent{agent_id}_x'
                agent_col_y = f'agent{agent_id}_y'
                
                if agent_col_x in traj_data.columns and agent_col_y in traj_data.columns:
                    agent_x = traj_data[agent_col_x].iloc[t]
                    agent_y = traj_data[agent_col_y].iloc[t]
                    
                    distance_to_origin = np.sqrt(agent_x**2 + agent_y**2)
                    
                    if distance_to_origin < min_distance_to_origin:
                        min_distance_to_origin = distance_to_origin
                        t_closest = t
                        closest_agent_id = agent_id
        #t_closest -= 2

        # Position of the closest agent at t_closest
        agent_col_x = f'agent{closest_agent_id}_x'
        agent_col_y = f'agent{closest_agent_id}_y'
        agent_x = traj_data[agent_col_x].iloc[t_closest]
        agent_y = traj_data[agent_col_y].iloc[t_closest]
        p_rec = np.array([agent_x, agent_y])
        """

        #Rcom_tx = compute_tx_communication_radius(p_rec, p_jammer)
        #Rcom_tx = min_distance_to_origin

    else:
        Rcom_tx = Rcom
    tx_circle = patches.Circle((0, 0), radius=Rcom_tx, color='blue', 
                            fill=False, linestyle='--', linewidth=1.5, 
                            alpha=0.5, label=r'$R_{\mathrm{com}}$')
    ax.add_patch(tx_circle)
    
    # Plot agent init area (circle with radius Ra centered at (R/2, 0))
    init_circle = patches.Circle((R/2, 0), radius=Ra, color='gray', 
                                 fill=False, linestyle=':', linewidth=1.5, 
                                 alpha=0.4)
    ax.add_patch(init_circle)
    
    # Plot jammer capsule area if jammer is on
    if jammer_on:
        cap_height = 3 * Rcom
        cap_radius = 1.5 * Rcom
        x_min = -cap_radius - 0.1 * Rcom
        x_max = R + cap_radius + 0.1 * Rcom
        
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
    else:
        x_min = -0.2 * Ra
        x_max = R + 0.2 * Ra
    
    # Plot all agent trajectories with time-dependent fading
    agent_label_added = False  # Track if we've added agent label to legend
    passive_agent_label_added = False  # Track if we've added passive agent label to legend
    
    for agent_id in range(1, k+1):
        agent_col_x = f'agent{agent_id}_x'
        agent_col_y = f'agent{agent_id}_y'
        
        if agent_col_x in traj_data.columns and agent_col_y in traj_data.columns:
            # Get initial and final positions
            initial_x = traj_data[agent_col_x].iloc[0]
            initial_y = traj_data[agent_col_y].iloc[0]
            final_x = traj_data[agent_col_x].iloc[-1]
            final_y = traj_data[agent_col_y].iloc[-1]
            
            is_passive = agent_id not in active_agents
            if is_passive:
                # Plot passive agent as a single black dot
                ax.scatter(initial_x, initial_y, s=150, alpha=0.8, edgecolors='darkgrey', 
                          linewidth=2, color=passive_color, marker='o', zorder=4)
                
                # Add passive agent label only once to legend
                if not passive_agent_label_added:
                    ax.plot([], [], alpha=0.8, linewidth=2, color=passive_color, marker='o', 
                           markersize=8, linestyle='none', label='Passive agent')
                    passive_agent_label_added = True
            else:
                # Plot initial position in grey
                ax.scatter(initial_x, initial_y, s=100, alpha=0.5, edgecolors='darkgrey', 
                          linewidth=1.5, color='grey', marker='o', zorder=3)
                
                # Plot trajectory segments with time-dependent alpha
                x_coords = traj_data[agent_col_x].values
                y_coords = traj_data[agent_col_y].values
                t_max = len(x_coords) - 1
                
                for t in range(len(x_coords) - 1):
                    # Alpha increases with time (0 at start, 1 at end)
                    alpha_val = 0.2 + 0.8 * (t / t_max)  # Range from 0.2 to 1.0
                    
                    ax.plot(x_coords[t:t+2], y_coords[t:t+2], 
                           alpha=alpha_val, linewidth=2, linestyle='--', color=agent_color, zorder=-1)
                
                # Plot final position as agent color dot
                ax.scatter(final_x, final_y, s=150, alpha=0.8, edgecolors='darkorange', 
                          linewidth=2, color=agent_color, marker='o', zorder=11)
                
                # Plot Rcom circle around final position ONLY for Baseline method
                if method.lower() == "baseline" and not directed_transmission:
                    if not jammer_on:  # isotropic, no jammer
                        Rcom_k = Rcom
                    elif jammer_on:  # isotropic, jammer (directed handled below)

                        # Compute time and position at which current agent stops moving
                        t_stop = len(traj_data) - 1
                        for t in range(len(traj_data)-1, 0, -1):
                            if (traj_data[agent_col_x].iloc[t] != traj_data[agent_col_x].iloc[t-1] or
                                traj_data[agent_col_y].iloc[t] != traj_data[agent_col_y].iloc[t-1]):
                                t_stop = t
                                break
                        p_agent_k = np.array([final_x, final_y])

                        # Compute jammer position at relay
                        if jammer_on:
                            jammer_x = traj_data['jammer_x'].iloc[t_stop] if 'jammer_x' in traj_data.columns else R/2
                            jammer_y = traj_data['jammer_y'].iloc[t_stop] if 'jammer_y' in traj_data.columns else 0.0
                            p_jammer = np.array([jammer_x, jammer_y])
                        else:
                            p_jammer = None
                        
                        Rcom_k = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=p_agent_k)
                        
                    agent_rcom_circle = patches.Circle((final_x, final_y), radius=Rcom_k, 
                                                    color=agent_color, fill=False, linestyle='--', 
                                                    linewidth=1.5, alpha=0.3)
                    ax.add_patch(agent_rcom_circle)
                
                # Add agent label only once to legend
                if not agent_label_added:
                    ax.plot([], [], alpha=0.6, linewidth=2, color=agent_color, label='Active agent trajectory')
                    agent_label_added = True
    
    # Plot transmitter (origin) in blue
    ax.scatter(0, 0, s=300, marker='s', color='blue', alpha=0.8, 
              edgecolors='darkblue', linewidth=2, label=r'$\mathrm{TX}$', zorder=5)
    
    # Plot receiver at (R, 0) in green
    ax.scatter(R, 0, s=300, marker='s', color='green', alpha=0.8, 
              edgecolors='darkgreen', linewidth=2, label=r'$\mathrm{RX}$', zorder=-1)

    # Plot antenna orientations at final positions if directed transmission
    if directed_transmission:
        if jammer_on:  # pick the last jammer position
            jammer_pos = np.array([traj_data['jammer_x'].iloc[-1], traj_data['jammer_y'].iloc[-1]])
        else:
            jammer_pos = None 

        for agent_id in range(1, k+1):
            agent_col_x = f'agent{agent_id}_x'
            agent_col_y = f'agent{agent_id}_y'
            agent_col_phi = f'agent{agent_id}_phi'
            
            if agent_col_x in traj_data.columns and agent_col_y in traj_data.columns and agent_col_phi in traj_data.columns:
                # Get final position and antenna direction
                final_x = traj_data[agent_col_x].iloc[-1]
                final_y = traj_data[agent_col_y].iloc[-1]
                final_phi = traj_data[agent_col_phi].iloc[-1]
                
                # Plot antenna lobe at final position
                plot_antenna_lobe(ax, np.array([final_x, final_y]), final_phi, Rcom, jammer_pos=jammer_pos, color=agent_color)

    
    # Plot jammer trajectory if present
    jammer_col_x = 'jammer_x'
    jammer_col_y = 'jammer_y'
    if jammer_col_x in traj_data.columns and jammer_col_y in traj_data.columns:
        # Filter out NaN values
        jammer_data = traj_data[[jammer_col_x, jammer_col_y]].dropna()
        if len(jammer_data) > 0:
            # Plot jammer trajectory with time-dependent fading
            jammer_x_coords = jammer_data[jammer_col_x].values
            jammer_y_coords = jammer_data[jammer_col_y].values
            jammer_t_max = len(jammer_x_coords) - 1
            
            for t in range(len(jammer_x_coords) - 1):
                # Alpha increases with time (0.2 at start, 1.0 at end)
                alpha_val = 0.2 + 0.8 * (t / jammer_t_max) if jammer_t_max > 0 else 0.6
                ax.plot(jammer_x_coords[t:t+2], jammer_y_coords[t:t+2], 
                    alpha=alpha_val, linewidth=2, linestyle='--', color='red', zorder=-1)
            
            # Plot final jammer position
            final_jammer_x = jammer_x_coords[-1]
            final_jammer_y = jammer_y_coords[-1]
            ax.scatter(final_jammer_x, final_jammer_y, s=200, marker='^', 
                    color='red', alpha=0.8, edgecolors='darkred', linewidth=2, label='Jammer', zorder=10)
            
    
    # Set axis properties
    ax.set_aspect('equal')
    #ax.set_xlabel(r'$x$', fontsize=12, fontweight='bold')
    #ax.set_ylabel(r'$y$', fontsize=12, fontweight='bold')

    # Remove frame and axis labels/ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set axis limits with margin
    all_x = [0, R]  # Include TX and RX
    all_y = [0]
    for agent_id in range(1, k+1):
        agent_col_x = f'agent{agent_id}_x'
        agent_col_y = f'agent{agent_id}_y'
        if agent_col_x in traj_data.columns:
            all_x.extend(traj_data[agent_col_x].dropna())
            all_y.extend(traj_data[agent_col_y].dropna())

    if jammer_col_x in traj_data.columns:
        all_x.extend(traj_data[jammer_col_x].dropna())
        all_y.extend(traj_data[jammer_col_y].dropna()) 

    if all_x and all_y:
        # Set x-limits to capsule endpoints with small margin
        cap_radius = 1.5 * Rcom if jammer_on else 0
        x_min = -cap_radius - 0.1 * Rcom
        x_max = R + cap_radius + 0.1 * Rcom
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1.1*Ra, 1.1*Ra)
    
    # Add legend and title
    ax.legend(loc='upper left')
    title = fr'Trajectory: {method} $K={k}$ Row={row}' + '\n' + \
            fr'directed$={int(directed_transmission)}$, jammer$={int(jammer_on)}$'
    #ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    ax.grid(False)

    # Save plot if plot_dir is provided
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}.pdf"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', transparent=False)
        print(f"Saved trajectory plot to {plot_path}")
    else:
        plt.show()



def animate_trajectory(traj_dir, k, row, directed_transmission, jammer_on, method="Baseline", anim_dir=None):
    """Fast animation using figure reuse and incremental updates."""
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    Rcom = 1.0
    agent_color_no_message = 'orange'
    agent_color_with_message = 'green'
    passive_color = 'black'
    
    traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
    traj_file_path = os.path.join(traj_dir, traj_filename)
    
    if not os.path.exists(traj_file_path):
        print(f"Trajectory file not found: {traj_file_path}")
        return
    
    traj_data = pd.read_csv(traj_file_path)
    print(f"Loaded trajectory from {traj_file_path}")
    print(f"Trajectory length: {len(traj_data)} time steps")
    
    R = traj_data['R'].iloc[0] if 'R' in traj_data.columns else 10.0
    Ra = 0.6 * R

    # Get list of active agents
    active_agents = []
    for agent_id in range(1, k+1):
        agent_col_has_message = f'agent{agent_id}_has_message'
        if np.any(traj_data[agent_col_has_message]):
            active_agents.append(agent_id)
    
    # Precompute agent passive status
    agent_passive_status = {}
    for agent_id in range(1, k+1):
        agent_col_x = f'agent{agent_id}_x'
        agent_col_y = f'agent{agent_id}_y'
        
        if agent_col_x in traj_data.columns and agent_col_y in traj_data.columns:
            initial_x = traj_data[agent_col_x].iloc[0]
            initial_y = traj_data[agent_col_y].iloc[0]
            final_x = traj_data[agent_col_x].iloc[-1]
            final_y = traj_data[agent_col_y].iloc[-1]
            
            is_passive = agent_id not in active_agents
            agent_passive_status[agent_id] = (is_passive, initial_x, initial_y, final_x, final_y)
    
    
    # ===== CREATE FIGURE ONCE =====
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.draw()
    
    # Precompute static elements
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(r'$x$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y$', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.2*Ra, R+0.2*Ra)
    ax.set_ylim(-1.1*Ra, 1.1*Ra)
    ax.grid(False)

    init_circle = patches.Circle((R/2, 0), radius=Ra, color='gray', 
                                fill=False, linestyle=':', linewidth=1.5, alpha=0.4)
    ax.add_patch(init_circle)
    
    if jammer_on:
        cap_height = 3 * Rcom
        cap_radius = 1.5 * Rcom
        ax.plot([0, R], [cap_height/2, cap_height/2], color='red', lw=2, alpha=0.4)
        ax.plot([0, R], [-cap_height/2, -cap_height/2], color='red', lw=2, alpha=0.4)
        theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
        ax.plot(cap_radius * np.cos(theta_left), cap_radius * np.sin(theta_left), 
               color='red', lw=2, alpha=0.4)
        theta_right = np.linspace(-np.pi/2, np.pi/2, 100)
        ax.plot(R + cap_radius * np.cos(theta_right), cap_radius * np.sin(theta_right), 
               color='red', lw=2, alpha=0.4)
    
    # Static elements (TX, RX)
    ax.scatter(0, 0, s=300, marker='s', color='blue', alpha=0.8, 
            edgecolors='darkblue', linewidth=2, zorder=5)
    ax.scatter(R, 0, s=300, marker='s', color='green', alpha=0.8, 
            edgecolors='darkgreen', linewidth=2, zorder=5)
    
    # Track number of static patches
    num_static_patches = len(ax.patches)
    num_static_lines = len(ax.get_lines())
    num_static_collections = len(ax.collections)
    
    pil_frames = []
    num_frames = len(traj_data)
    dur_frame = 5
    
    print(f"Generating {num_frames} frames...")
    print(f"Static elements: {num_static_patches} patches, {num_static_lines} lines, {num_static_collections} collections")
    
    for t in range(num_frames):
        # Clear ONLY dynamic content
        # Remove all lines added after the static ones
        while len(ax.get_lines()) > num_static_lines:
            ax.get_lines()[-1].remove()
        
        # Remove all patches added after the static ones
        while len(ax.patches) > num_static_patches:
            ax.patches[-1].remove()
        
        # Remove all collections added after the static ones
        while len(ax.collections) > num_static_collections:
            ax.collections[-1].remove()

        # Get jammer info if present
        jammer_col_x = 'jammer_x'
        jammer_col_y = 'jammer_y'
        p_jammer = None
        if jammer_col_x in traj_data.columns:
            jammer_data = traj_data[[jammer_col_x, jammer_col_y]].iloc[:t+1].dropna()
            if len(jammer_data) > 0:
                final_jammer_x = jammer_data[jammer_col_x].iloc[-1]
                final_jammer_y = jammer_data[jammer_col_y].iloc[-1]
                p_jammer = np.array([final_jammer_x, final_jammer_y])
                
                # Plot jammer trajectory
                ax.plot(jammer_data[jammer_col_x], jammer_data[jammer_col_y], 
                       alpha=0.6, linewidth=2, linestyle='--', color='red', zorder=0)
                
                # Plot jammer position
                ax.scatter(final_jammer_x, final_jammer_y, s=200, marker='^', 
                          color='red', alpha=0.8, edgecolors='darkred', linewidth=2, zorder=6)
        
        # Add TX communication range circle (dynamic, updates each frame based on jammer)
        if jammer_on and p_jammer is not None:
            Rcom_tx = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                        jammer_pos=p_jammer, p_tx=np.array([0, 0]))
        else:
            Rcom_tx = Rcom
        
        tx_rcom_circle = patches.Circle((0, 0), radius=Rcom_tx, color='blue', 
                                        fill=False, linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
        ax.add_patch(tx_rcom_circle)
        
        # Draw trajectory paths (dynamic)
        for agent_id in range(1, k+1):
            agent_col_x = f'agent{agent_id}_x'
            agent_col_y = f'agent{agent_id}_y'
            agent_col_phi = f'agent{agent_id}_phi'
            agent_col_has_message = f'agent{agent_id}_has_message'
            
            if agent_col_x not in traj_data.columns:
                continue
            
            is_passive, init_x, init_y, final_x, final_y = agent_passive_status[agent_id]
            
            # Check if agent has message at current time
            has_message_current = False
            if agent_col_has_message in traj_data.columns:
                has_message_current = bool(traj_data[agent_col_has_message].iloc[t])
            
            # Determine agent color based on message status
            agent_color = agent_color_with_message if has_message_current else agent_color_no_message
            
            if is_passive:
                ax.scatter(init_x, init_y, s=150, alpha=0.8, edgecolors='darkgrey', 
                        linewidth=2, color=passive_color, marker='o', zorder=4)
            else:
                x_coords = traj_data[agent_col_x].values[:t+1]
                y_coords = traj_data[agent_col_y].values[:t+1]
                
                if len(x_coords) > 1:
                    # Plot path segments
                    for time_idx in range(len(x_coords) - 1):
                        alpha_val = 0.2 + 0.8 * (time_idx / max(len(x_coords) - 1, 1))
                        ax.plot(x_coords[time_idx:time_idx+2], y_coords[time_idx:time_idx+2], 
                            alpha=alpha_val, linewidth=2, linestyle='--', color=agent_color, zorder=0)
                
                # Current position
                ax.scatter(x_coords[-1], y_coords[-1], s=150, alpha=0.8, 
                        edgecolors=agent_color if has_message_current else 'darkorange', 
                        linewidth=2, color=agent_color, 
                        marker='o', zorder=4)
                
                # Plot Rcom circle for baseline method (only if not directed transmission)
                if method.lower() == "baseline" and not directed_transmission:
                    if jammer_on and p_jammer is not None:
                        Rcom_k = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=np.array([x_coords[-1], y_coords[-1]]))
                    else:
                        Rcom_k = Rcom
                    agent_rcom = patches.Circle((x_coords[-1], y_coords[-1]), 
                                            radius=Rcom_k, color=agent_color, fill=False, 
                                            linestyle='--', linewidth=1.5, alpha=0.3, zorder=2)
                    ax.add_patch(agent_rcom)
                
                # Plot antenna lobe at current position if directed transmission
                if directed_transmission and agent_col_phi in traj_data.columns:
                    current_phi = traj_data[agent_col_phi].iloc[t]
                    if jammer_on:
                        plot_antenna_lobe(ax, np.array([x_coords[-1], y_coords[-1]]), current_phi, Rcom, 
                                          jammer_pos=p_jammer, color=agent_color)
                    else:
                        plot_antenna_lobe(ax, np.array([x_coords[-1], y_coords[-1]]), current_phi, Rcom, jammer_pos=None, color=agent_color)
        
        # Update title only
        title = fr'Trajectory: {method} $K={k}$, row={row}' + f', $t = {t}$' + '\n' + \
                fr'directed$={int(directed_transmission)}$, jammer$={int(jammer_on)}$'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Draw and capture
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image_array = np.frombuffer(buf, dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_rgb = image_array[:, :, :3]
        
        pil_frames.append(Image.fromarray(image_rgb, 'RGB'))
        
        if (t + 1) % max(1, num_frames // 10) == 0 or t == 0:
            print(f"  Generated frame {t+1}/{num_frames}")

        ax.grid(False)
    
    plt.close(fig)
    
    # Save GIF
    if anim_dir:
        os.makedirs(anim_dir, exist_ok=True)
        gif_path = os.path.join(anim_dir, 
                               f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}.gif")
        
        print(f"\nSaving animation to {gif_path}...")
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], 
                          duration=dur_frame, loop=0)
        
        print(f"✓ Saved animation to {gif_path}")
        print(f"  Total frames: {len(pil_frames)}")
        print(f"  Duration per frame: {dur_frame}ms")
    else:
        print("Animation frames generated (not saved)")


def main():
    # Choose evaluation mode
    # (USE) 0-generate baseline result data  
    # (USE) 1-evaluate single policy  
    # 2-compare baseline policies 
    # 3-examine specific instances 
    # (USE DEBUG) 4-plot/animate solutions of specific instances
    # (USE DEBUG) 5-plot init scenario
    # 6-compare scenarios for one specific policy
    # 7-compare multiple policies simultaneously
    # 8-save trajectory from specific scenario
    # (USE) 9-evaluate specified methods on a specific scenario  
    # (USE) 10-compare baseline and MADDPG or MAPPO policy  
    # 11-violin plot of value, delivery time, and total distance (maybe use SMAPE) vs K
    # (USE)12-generate trajectory data from the baseline (runs Dijkstra)
    # (USE)13-plot trajectory (takes trajectory file)
    # (USE)14-animation trajectory (takes trajectory file)
    # 15-compare mappo and maddpg against each other
    # (USE) 16-heatmaps over all K and methods simultaneously for one scenario
    # (TODO)17-heatmaps over all K simultaneously for all scenarios but only value (4x3 subplots)
    # (TODO)18-for a trajectory, plot the resulting policies  (3x1)
    # (TODO)19- -||- animation results  (3x1)
    # 20-compare baseline values for two various K just to see how much the value distribution differs
    # 21-make a 2x2 (4x1?) plot for baseline comparing trajectory between all scenarios for a specific game instance
    # 22-make a 2x2 animation for baseline for specific scenario

    
    testing = False  # are we running on test data? (FINAL DATA) False -> Evaluation data

    eval_mode = 14

    eval_K = [5] 
    value_remove_below = -10  
    value_remove_above = 20
    time_remove_below = -0  
    time_remove_above = 65 
    dist_remove_below = 0  
    dist_remove_above = 100 
    """
    value_remove_below = 0  
    value_remove_above = 4 
    time_remove_below = -0  
    time_remove_above = 65 
    dist_remove_below = 0  
    dist_remove_above = 40 
    """
    
    present_mode = "violin"  # hist ; violin
    compare_mode = "violin"  # hist ; scatter ; heatmap; violin

    # Configuration
    K_start = 1
    K_end = 10
    K = range(K_start, K_end+1)
    row_start = 1  # +1 of row_idx  # K=20 -> 1797
    row_end = 10000  # Specify the range of rows to evaluate

    c_pos = 0.5 #[0.5, 1]  # motion cost parameter
    c_phi = 0.1  # antenna cost parameter
    
    directed_transmission = True 
    jammer_on = False
    clustering_on = True 
    minimize_distance = False  # minimize total movement instead of fasted delivery time

    # Configuration string for saving/loading
    conf_string = get_config_string(directed_transmission, jammer_on, c_pos=c_pos, c_phi=c_phi)

    # Compare baseline methods. Choose corresponding evaluation folders
    method1 = "baseline_nocluster"
    method2 = "baseline"
    method3 = "baseline_directed"
    if method1 == "baseline_nocluster":
        eval_folder1 = get_config_string(directed_transmission=False, jammer_on=False, c_pos=c_pos, c_phi=c_phi)
    if method2 == "baseline":
        eval_folder2 = get_config_string(directed_transmission=False, jammer_on=False, c_pos=c_pos, c_phi=c_phi)
    if method3 == "baseline_directed":
        eval_folder3 = get_config_string(directed_transmission=True, jammer_on=False, c_pos=c_pos, c_phi=c_phi)

    # Compare baseline with MARL
    baseline_eval_folder = get_config_string(directed_transmission=directed_transmission, jammer_on=jammer_on, c_pos=c_pos, c_phi=c_phi)
    marl_eval_folder = "Evaluation/MARL_evaluations"
    marl_alg = "MAPPO"  # MADDPG or MAPPO
    marl_file = get_marl_config_string(marl_alg, K_start, c_pos, c_phi, directed_transmission, jammer_on)

    # Loading initial scenarios to evaluate/test on
    if testing:
        data_dir = "Testing/Data/Test_states"
        data_files = ["test_states_K" + str(k) + "_n10000.csv" for k in K]  # Specify your data files
    else:
        data_dir = "Evaluation/Evaluation_states/Data"
        data_files = ["evaluation_states_K" + str(k) + "_n10000.csv" for k in K]  # Specify your data files
    data_file_paths = [data_dir + "/" + data_file for data_file in data_files]
    rows = range(row_start-1, row_end)  # Specify rows to evaluate
    
    # Saving stuff
    if testing:
        plot_dir = f"Testing/Plots/{conf_string}"
        result_dir = "Testing/Data/Results"
    else:  # evaluating
        plot_dir = f"Plots/Evaluation_plots/{conf_string}"
        result_dir = "Evaluation/Evaluation_results"
    os.makedirs(plot_dir, exist_ok=True)

    if eval_mode == 0:  # Generate evaluation data
        eval_string = "Testing" if testing else "Evaluation"
        for i, k in enumerate(K):
            print(f"\n{eval_string} baseline with K={k}, directed_transmission={directed_transmission}, jammer_on={jammer_on}, clustering_on={clustering_on}, minimize_distance={minimize_distance}")
            data_file_path = data_file_paths[i]

            # Evaluation results
            result = evaluate_baseline(data_file_path, c_pos, c_phi, rows, directed_transmission=directed_transmission, jammer_on=jammer_on, clustering_on=clustering_on, minimize_distance=minimize_distance)

            # Save evaluation results
            if result_dir:
                n = rows[-1]+1
                save_evaluation_results(
                    result, result_dir, k, c_pos, c_phi, n,
                    directed_transmission=directed_transmission,
                    jammer_on=jammer_on,
                    clustering_on=clustering_on,
                    minimize_distance=minimize_distance
                )
    elif eval_mode == 1:  # Evaluate single policy
        # Read evaluation results
        if present_mode == "hist":
            for i, k in enumerate(K):
                eval_data_file_path = os.path.join(result_dir, conf_string, f"evaluation_results_K{k}_n{row_end}_{conf_string}.csv")
                data = pd.read_csv(eval_data_file_path)
                results = data.to_dict(orient='records')

                
                # Plot histograms
                print(f"Result: {len(results)} evaluations")
                plot_histogram(results, plot_dir=plot_dir, c_pos=c_pos, c_phi=c_phi)
        elif present_mode == "violin":  # 3x1 three subplots of metrics vs K
            results_dict = {}
            for i, k in enumerate(K):
                eval_data_file_path = os.path.join(result_dir, conf_string, f"evaluation_results_K{k}_n{row_end}_{conf_string}.csv")
                data = pd.read_csv(eval_data_file_path)
                results = data.to_dict(orient='records')
                results_dict[k] = results

            plot_violin_plots(results_dict, plot_dir=plot_dir, c_pos=c_pos, c_phi=c_phi)
    elif eval_mode == 2:  # Compare policies
        for i, k in enumerate(K):
            # For every K, read both evaluation results and compute difference in performance measures and plot diff histograms
            eval_data_file_path1 = os.path.join(result_dir, eval_folder1, f"evaluation_results_K{k}_n{row_end}_{eval_folder1}.csv")
            data1 = pd.read_csv(eval_data_file_path1)
            results1 = data1.to_dict(orient='records')

            eval_data_file_path2 = os.path.join(result_dir, eval_folder2, f"evaluation_results_K{k}_n{row_end}_{eval_folder2}.csv")
            data2 = pd.read_csv(eval_data_file_path2)
            results2 = data2.to_dict(orient='records')

            # Actual comparison
            if len(results1) != len(results2):
                print(f"Warning: Different number of results for K={k}. results1: {len(results1)}, results2: {len(results2)}")
                # Use the minimum length to avoid index errors
                min_len = min(len(results1), len(results2))
                results1 = results1[:min_len]
                results2 = results2[:min_len]
            
            # Compute differences for each measure (results2 - results1)
            differences = []
            metrics = ['value', 'delivery_time', 'agent_sum_distance', 'message_air_distance']
            
            if len(results1) != len(results2):
                raise RuntimeError("Mismatched result lengths.")

            nr_points = len(results1)
            for j in range(nr_points):  
                diff_dict = {
                    'idx': results1[j]['idx'],
                    'K': k,
                    'value_diff': results2[j]['value'] - results1[j]['value'],
                    'delivery_time_diff': results2[j]['delivery_time'] / results1[j]['delivery_time'],
                    'agent_sum_distance_diff': results2[j]['agent_sum_distance'] - results1[j]['agent_sum_distance'],
                    'message_air_distance_diff': results2[j]['message_air_distance'] / results1[j]['message_air_distance']
                }
                differences.append(diff_dict)

            if compare_mode == "hist":
                compare_plot_dir = f"Plots/Evaluation_plots/comparisons/{method1}_vs_{method2}"
                #plot_comparison_histograms(differences, method1, method2, 
                #                            plot_dir=compare_plot_dir, k=k)
            elif compare_mode == "scatter":
                plot_comparison_scatter(results1, results2, method1, method2, 
                                            plot_dir=None, k=k)
            elif compare_mode == "heatmap":
                plot_comparison_heatmap(results1, results2, method1, method2, 
                                            plot_dir=None, k=k)
                

            
            # Print summary statistics
            print(f"\nK={k} Comparison Summary ({method2} - {method1}):")
            for metric in metrics:
                diff_values = [d[f'{metric}_diff'] for d in differences]
                mean_diff = np.mean(diff_values)
                std_diff = np.std(diff_values)
                print(f"  {metric}: mean={mean_diff:.4f}, std={std_diff:.4f}")
    
    elif eval_mode == 3:  # Examine specific instances
        # Read evaluation results
        for i, k in enumerate(K):
            eval_data_file_path = os.path.join(result_dir, conf_string, f"evaluation_results_K{k}_n{row_end}_{conf_string}.csv")
            eval_data = pd.read_csv(eval_data_file_path)
            results = eval_data.to_dict(orient='records')

            rows_below_zero = get_rows_by_value(results, 0.0, mode='lower')
            print(rows_below_zero)

            # Rows that are negative without clustering
            if k == 2:
                rows_below_zero = [1272, 1475, 2785, 2958, 3122, 3680, 4000, 4991, 8408, 8681, 9692]
            elif k == 3:
                rows_below_zero = [2964, 4477, 6085, 6575, 8836]

            # Read data files again to get state dicts
            data_file = data_file_paths[i]
            data = pd.read_csv(data_file)

            for row in rows_below_zero:
                row_idx = row - 1  
                state = get_state_dict(data, row_idx)

                baseline_result = baseline(
                    state['p_agents'], 
                    state['p_tx'],
                    state['p_recv'],
                    Rcom=state['Rcom'],
                    sigma=state['sigma'],
                    beta=state['beta'],
                    c_pos=c_pos,
                    directed_transmission=directed_transmission,
                    jammer=jammer_on,
                    clustering=clustering_on
                )

                savepath = f"Plots/Evaluation_plots/{conf_string}/Negative_instances/K{k}_row{row}.png"
                os.makedirs(os.path.dirname(savepath), exist_ok=True)

                # OBSOLETE, USE plot_trajectory INSTEAD
                plot_scenario_with_path_colored(state, baseline_result, 
                                                savepath=savepath, jammer=False, debug=False)
    
    elif eval_mode == 4:  # Plot specific instances
        # Read evaluation results  # K=4 row=14 receiving agent moves suspiciously far (animation can be improved?)
        row_idx = row_start-1  # K2 row 30 directed strange
        k = K_start
        data = pd.read_csv(data_file_paths[0])
        state = get_state_dict(data, row_idx)
        #plot_scenario(state, directed=False, jammer=False, savepath=None, debug=True)
        #directed_transmission = None
        only_init = True
        c_phi = 0.1

        if only_init:
            plot_scenario(state, directed=directed_transmission, jammer=jammer_on, savepath=None, debug=True)
        else:
            baseline_result = baseline(
                        state['p_agents'], 
                        state['p_tx'],
                        state['p_recv'],
                        Rcom=state['Rcom'],
                        sigma=state['sigma'],
                        beta=state['beta'],
                        c_pos=c_pos,
                        c_phi=c_phi,
                        phi_agents=state['phi_agents'] if directed_transmission else None,
                        jammer_info=state['jammer_info'] if jammer_on else None,
                        clustering=True
                    )

            #plot_scenario_with_path_colored(state, baseline_result, 
            #                                        savepath=None, directed_transmission=directed_transmission, jammer=False, debug=True)
            
            savepath = f"Animations/experimentation/{conf_string}/K{k}_row{row_idx+1}.gif"
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            #savepath = None
            #animate_scenario_with_path_colored_10nov(state, baseline_result, directed_transmission=directed_transmission, jammer_on=jammer_on,
            #                                savepath=savepath, beamer_gif=False, interval=100)
    
    elif eval_mode == 5:  # Plot specific instance initial scenario
    
        row_idx = row_start-1
        k = 1
        if testing:
            data_name = "test_states_K{k}_n10000.csv"
        else:
            data_name = "evaluation_states_K{k}_n10000.csv"
        data_file_path = data_dir + "/" + data_name
        data = pd.read_csv(data_file_path)
        state = get_state_dict(data, row_idx)
        #plot_scenario(state, directed=directed_transmission, jammer=False, savepath=None, debug=True)

        baseline_result = baseline(
                    state['p_agents'], 
                    state['p_tx'],
                    state['p_recv'],
                    Rcom=state['Rcom'],
                    sigma=0.2,
                    beta=state['beta'],
                    c_pos=c_pos,
                    c_phi=c_phi,
                    phi_agents=state['phi_agents'] if directed_transmission else None,
                    jammer_info=state['jammer_info'] if jammer_on else None,
                    clustering=True
        )
        
        # OBSOLETE, USE plot_trajectory INSTEAD
        plot_scenario_with_path_colored(state, baseline_result, savepath=None, directed_transmission=False, jammer=False, debug=False)
    
    elif eval_mode == 7:  # Evaluate and compare multiple policies simultaneously
        # Load the evaluation data files of interest
        k = K_start

        # Define method configurations
        methods = [method1, method2, method3]
        eval_folders = [eval_folder1, eval_folder2, eval_folder3]
        
        # Load all results into a dictionary
        results_dict = {}
        test_string = "test" if testing else "evaluation"
        for method, eval_folder in zip(methods, eval_folders):
            eval_data_file_path = os.path.join(result_dir, eval_folder, f"{test_string}_results_K{k}_n{row_end}_{eval_folder}.csv")
            data = pd.read_csv(eval_data_file_path)
            results_dict[method] = data.to_dict(orient='records')
        
        # Plot heatmap matrix
        if testing:
            compare_plot_dir = f"Testing/Plots/Comparison"
        else:
            compare_plot_dir = f"Plots/Evaluation_plots/comparisons"
        plot_comparison_heatmap_matrix(results_dict, methods, plot_dir=compare_plot_dir, k=k)
        
        # Also perform pairwise comparisons if desired
        #plot_all_pairwise_comparisons(results_dict, methods, compare_mode=compare_mode, 
        #                     plot_dir=compare_plot_dir, k=k)              
    
    elif eval_mode == 8:  # Save trajectory from specific scenario
        # OBS! Assumes directed transmission
        row_idx = row_start-1
        k = K_start
        data = pd.read_csv(data_file_paths[0])
        print(f'Loaded {data_file_paths[0]}')
        state = get_state_dict(data, row_idx)

        baseline_result = baseline(
                    state['p_agents'], 
                    state['p_tx'],
                    state['p_recv'],
                    Rcom=state['Rcom'],
                    sigma=0.2,
                    beta=state['beta'],
                    c_pos=c_pos,
                    c_phi=c_phi,
                    phi_agents=state['phi_agents'] if directed_transmission else None,
                    jammer_info=state['jammer_info'] if jammer_on else None,
                    clustering=True
                )
        
        # Convert trajectory dict to csv and save
        savepath = f"Evaluation/Evaluation_states/Trajectories/K{k}_row{row_start}_trajectory.csv"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        save_trajectory(baseline_result, savepath=savepath)
    
    elif eval_mode == 9:  # Compare baseline and MARL policies
        if compare_mode == "hist":
            # Read evaluation results
            eval_data_file_path = f'{marl_eval_folder}/{marl_alg}/{marl_file}.csv'
            data = pd.read_csv(eval_data_file_path, index_col=False)
            #data['K'] = data['file']  # temporary fix
            result_marl = data.to_dict(orient='records')

            # Plot histograms
            eval_plot_dir = f"{marl_eval_folder}/{marl_alg}/Plots"
            os.makedirs(eval_plot_dir, exist_ok=True)
            plot_histogram(result_marl, plot_dir=eval_plot_dir, c_pos=c_pos, c_phi=c_phi, method=marl_alg)
        elif compare_mode == "violin":  # 3x1 three subplots of metrics vs K
            baseline_results_dict = {}  
            maddpg_results_dict = {}
            for k in eval_K:
                # Load result data
                eval_data_file_path_baseline = os.path.join(result_dir, conf_string, f"evaluation_results_K{k}_n{row_end}_{conf_string}.csv")
                baseline_data = pd.read_csv(eval_data_file_path_baseline)
                eval_data_file_path_maddpg = f'{marl_eval_folder}/MADDPG/{marl_file}.csv'
                maddpg_data = pd.read_csv(eval_data_file_path_maddpg)

                # Remove outliers based on value 
                # Count number of removed entries
                num_removed_baseline = len(baseline_data) - len(baseline_data[baseline_data['value'] >= value_remove_below])
                num_removed_maddpg = len(maddpg_data) - len(maddpg_data[maddpg_data['value'] >= value_remove_below])
                print(f"K={k}: Removed {num_removed_baseline} baseline entries and {num_removed_maddpg} MADDPG entries below value {value_remove_below}")
                baseline_data = baseline_data[baseline_data['value'] >= value_remove_below].reset_index(drop=True)
                maddpg_data = maddpg_data[maddpg_data['value'] >= value_remove_below].reset_index(drop=True)
                
                baseline_results = baseline_data.to_dict(orient='records')
                baseline_results_dict[k] = baseline_results
                maddpg_results = maddpg_data.to_dict(orient='records')
                maddpg_results_dict[k] = maddpg_results

            eval_plot_dir = f"Plots/Evaluation_plots/comparisons/compare_violin_plots_dir{int(directed_transmission)}_jam{int(jammer_on)}_cpos{c_pos}_cphi{c_phi}"
            os.makedirs(eval_plot_dir, exist_ok=True)

            result_dicts = {
                "baseline": baseline_results_dict,
                "MADDPG": maddpg_results_dict   
            }
            plot_violin_plots(result_dicts, plot_dir=eval_plot_dir, c_pos=c_pos, c_phi=c_phi, directed_transmission=directed_transmission, jammer_on=jammer_on)
    
    elif eval_mode == 10:  # Compare baseline and maddpg or mappo policy
        # Baseline results
        eval_data_file_path_baseline = os.path.join(result_dir, baseline_eval_folder, f"evaluation_results_K{K_start}_n{row_end}_{baseline_eval_folder}.csv")
        data_baseline = pd.read_csv(eval_data_file_path_baseline)
        results_baseline = data_baseline.to_dict(orient='records')
        
        # MARL results
        eval_data_file_path_marl = f'{marl_eval_folder}/{marl_alg}/{marl_file}.csv'
        data_marl = pd.read_csv(eval_data_file_path_marl, index_col=False)
        result_marl = data_marl.to_dict(orient='records')

        # Remove specific problematic row if exists  (K=1:5393, K=5:7865, K=7:1411)
        #data_marl = data_marl.drop(1411).reset_index(drop=True)

        # Sanity check, results of same length and has the same R and budget
        try:
            if len(results_baseline) != len(result_marl):
                raise RuntimeError(f"Mismatch in number of results: baseline {len(results_baseline)} vs {marl_alg} {len(result_marl)}")
            else:
                for i in range(len(results_baseline)):
                    if results_baseline[i]['R'] != result_marl[i]['R'] or results_baseline[i]['budget'] != result_marl[i]['budget']:
                        raise RuntimeError(f"Mismatch in R or budget at index {i}: baseline R={results_baseline[i]['R']}, budget={results_baseline[i]['budget']} vs {marl_alg} R={result_marl[i]['R']}, budget={result_marl[i]['budget']}")
        except RuntimeError as e:
            print(f"Potential fail: {e}")
            # Examine which element is off
            for i in range(min(len(results_baseline), len(result_marl))):
                if not np.isclose(results_baseline[i]['R'], result_marl[i]['R'], atol=1e-6) or not np.isclose(results_baseline[i]['budget'], result_marl[i]['budget'], atol=1e-6):
                    
                    raise RuntimeError(f"Mismatch at index {i}: baseline R={results_baseline[i]['R']}, budget={results_baseline[i]['budget']} vs {marl_alg} R={result_marl[i]['R']}, budget={result_marl[i]['budget']}")


        # Plot histograms
        eval_plot_dir = f"{marl_eval_folder}/Compare/{marl_alg}_vs_baseline"
        os.makedirs(eval_plot_dir, exist_ok=True)

        plot_comparison_heatmap(results_baseline, result_marl, "baseline", marl_alg, 
                                            plot_dir=eval_plot_dir, k=K_start)
    
    elif eval_mode == 13:  # Plot trajectory (takes trajectory file)

        k = 5
        row = 1
        if k not in range(1,11):
            no_metrics = True
        else:
            no_metrics = False
        #jammer_on=False
        method = "Baseline"  # Baseline or MADDPG or MAPPO
        force_gen_traj = True

        if testing:
            traj_dir = f"Testing/Data/Trajectories/{method}"
            data_dir = "Testing/Data/Test_states"
            plot_dir = f"Testing/Plots/{method}/Trajectories/{conf_string}"
        else:
            traj_dir = f"Evaluation/Trajectories/{method}"
            data_dir = "Evaluation/Evaluation_states/Data"
            plot_dir = f"Evaluation/Plots/{method}/Trajectories/{conf_string}"

        # Construct trajectory file name to check if it exists
        traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
        traj_file_path = os.path.join(traj_dir, traj_filename)
        
        # If trajectory doesn't exist and method is Baseline, generate it
        if not os.path.exists(traj_file_path) or force_gen_traj:
            print(f"Trajectory file not found: {traj_file_path}")
            if method.lower() == "baseline":
                print(f"Generating baseline trajectory...")
                generate_baseline_trajectory(k, row, data_dir, c_pos, c_phi, 
                                            directed_transmission=directed_transmission, 
                                            jammer_on=jammer_on, 
                                            testing=testing, no_metrics=no_metrics)
            else:
                print(f"Trajectory file not found and method is not Baseline. Cannot generate.")
        
        # Plot the trajectory
        plot_trajectory(traj_dir, k, row, directed_transmission, jammer_on, method=method, plot_dir=plot_dir)
    
    elif eval_mode == 14:  # Plot trajectory (takes trajectory file)

        k = 1
        row = 15
        #directed_transmission = False
        method = "Baseline"  # Baseline or MADDPG or MAPPO
        force_gen_traj = True

        if testing:
            traj_dir = f"Testing/Data/Trajectories/{method}"
            data_dir = "Testing/Data/Test_states"
            anim_dir = f"Testing/Animation/{method}/Trajectories/{conf_string}"
        else:
            traj_dir = f"Evaluation/Trajectories/{method}"
            data_dir = "Evaluation/Evaluation_states/Data"
            anim_dir = f"Evaluation/Animation/{method}/Trajectories/{conf_string}"

        # Construct trajectory file name to check if it exists
        traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
        traj_file_path = os.path.join(traj_dir, traj_filename)
        
        # If trajectory doesn't exist and method is Baseline, generate it
        if not os.path.exists(traj_file_path) or force_gen_traj:
            print(f"Trajectory file not found: {traj_file_path}")
            if method.lower() == "baseline":
                print(f"Generating baseline trajectory...")
                generate_baseline_trajectory(k, row, data_dir, c_pos, c_phi, 
                                            directed_transmission=directed_transmission, 
                                            jammer_on=jammer_on, 
                                            testing=testing)
            else:
                print(f"Trajectory file not found and method is not Baseline. Cannot generate.")
        
        # Animate the trajectory
        animate_trajectory(traj_dir, k, row, directed_transmission, jammer_on, method=method, anim_dir=anim_dir)
    
    elif eval_mode == 15:  # Compare MAPPO and MADDPG
        # Determine which algorithm folders to use
        mappo_alg = "MAPPO"
        maddpg_alg = "MADDPG"
        
        if testing:
            marl_eval_folder = "Testing/Data/Results"
        else:
            marl_eval_folder = "Evaluation/MARL_evaluations"
        
        # Load MAPPO results
        mappo_file = get_marl_config_string(mappo_alg, K_start, c_pos, c_phi, directed_transmission, jammer_on, testing=testing)
        eval_data_file_path_mappo = os.path.join(marl_eval_folder, mappo_alg, f"{mappo_file}.csv")
        
        if not os.path.exists(eval_data_file_path_mappo):
            print(f"MAPPO file not found: {eval_data_file_path_mappo}")
        else:
            data_mappo = pd.read_csv(eval_data_file_path_mappo)
        
        # Load MADDPG results
        maddpg_file = get_marl_config_string(maddpg_alg, K_start, c_pos, c_phi, directed_transmission, jammer_on, testing=testing)
        eval_data_file_path_maddpg = os.path.join(marl_eval_folder, maddpg_alg, f"{maddpg_file}.csv")
        
        if not os.path.exists(eval_data_file_path_maddpg):
            print(f"MADDPG file not found: {eval_data_file_path_maddpg}")
        else:
            data_maddpg = pd.read_csv(eval_data_file_path_maddpg)

        # Filter rows where BOTH algorithms have value >= value_remove_below
        # This ensures the same data points are kept for both
        valid_indices_mappo = data_mappo[data_mappo['value'] >= value_remove_below].index
        valid_indices_maddpg = data_maddpg[data_maddpg['value'] >= value_remove_below].index
        
        # Keep only rows that are valid in BOTH datasets
        valid_indices = valid_indices_mappo.intersection(valid_indices_maddpg)
        
        data_mappo_filtered = data_mappo.loc[valid_indices].reset_index(drop=True)
        data_maddpg_filtered = data_maddpg.loc[valid_indices].reset_index(drop=True)
        
        num_removed = len(data_mappo) - len(data_mappo_filtered)
        
        results_mappo = data_mappo_filtered.to_dict(orient='records')
        results_maddpg = data_maddpg_filtered.to_dict(orient='records')
        
        
        # Sanity check: results should have same length and matching R and budget
        try:
            if len(results_mappo) != len(results_maddpg):
                raise RuntimeError(f"Mismatch in number of results: MAPPO {len(results_mappo)} vs MADDPG {len(results_maddpg)}")
            else:
                for i in range(len(results_mappo)):
                    if not np.isclose(results_mappo[i]['R'], results_maddpg[i]['R'], atol=1e-6) or \
                    not np.isclose(results_mappo[i]['budget'], results_maddpg[i]['budget'], atol=1e-6):
                        raise RuntimeError(f"Mismatch at index {i}: MAPPO R={results_mappo[i]['R']}, budget={results_mappo[i]['budget']} vs MADDPG R={results_maddpg[i]['R']}, budget={results_maddpg[i]['budget']}")
        except RuntimeError as e:
            print(f"Error: {e}")
            raise
            
        eval_plot_dir = os.path.join(marl_eval_folder, "Compare", f"{maddpg_alg}_vs_{mappo_alg}")
        os.makedirs(eval_plot_dir, exist_ok=True)
        plot_comparison_heatmap(results_mappo, results_maddpg, mappo_alg, maddpg_alg, 
                            plot_dir=eval_plot_dir, k=K_start, detailed=True)
    
    elif eval_mode == 16:  # Heatmaps over all K simultaneously for one scenario
        """
        Load results for selected methods across all K values.
        Compute pairwise differences and plot heatmaps for all K in one figure per comparison.
        """
        
        # ===== SELECT WHICH METHODS TO COMPARE =====
        methods_to_compare = ["baseline",  "MAPPO"]  # Choose subset: e.g., ["baseline", "MADDPG"]
        # Other options:
        # methods_to_compare = ["baseline", "MADDPG"]
        # methods_to_compare = ["baseline", "MAPPO"]
        # methods_to_compare = ["MADDPG", "MAPPO"]
        # methods_to_compare = ["baseline"]
        
        # Determine which result folders to use
        if testing:
            marl_eval_folder = "Testing/Data/Results"
            baseline_result_folder = "Testing/Data/Results"
        else:
            marl_eval_folder = "Evaluation/MARL_evaluations"
            baseline_result_folder = "Evaluation/Evaluation_results"
        
        baseline_eval_folder = get_config_string(directed_transmission, jammer_on, c_pos=c_pos, c_phi=c_phi)
        
        # Load results for selected methods only
        results_dicts = {}
        
        if "baseline" in methods_to_compare:
            print("Loading baseline results...")
            baseline_results_dict = {}
            for k in eval_K:
                eval_data_file_path = os.path.join(baseline_result_folder, baseline_eval_folder, 
                                                f"evaluation_results_K{k}_n{row_end}_{baseline_eval_folder}.csv")
                if os.path.exists(eval_data_file_path):
                    data = pd.read_csv(eval_data_file_path)
                    baseline_results_dict[k] = data.to_dict(orient='records')
                    print(f"  Loaded baseline K={k}: {len(baseline_results_dict[k])} entries")
                else:
                    print(f"  Warning: Baseline file not found for K={k}: {eval_data_file_path}")
            results_dicts["baseline"] = baseline_results_dict
        
        if "MADDPG" in methods_to_compare:
            print("Loading MADDPG results...")
            maddpg_results_dict = {}
            for k in eval_K:
                maddpg_file = get_marl_config_string("MADDPG", k, c_pos, c_phi, directed_transmission, jammer_on, testing=testing)
                eval_data_file_path = os.path.join(marl_eval_folder, "MADDPG", f"{maddpg_file}.csv")
                if os.path.exists(eval_data_file_path):
                    data = pd.read_csv(eval_data_file_path)
                    maddpg_results_dict[k] = data.to_dict(orient='records')
                    print(f"  Loaded MADDPG K={k}: {len(maddpg_results_dict[k])} entries")
                else:
                    print(f"  Warning: MADDPG file not found for K={k}: {eval_data_file_path}")
            results_dicts["MADDPG"] = maddpg_results_dict
        
        if "MAPPO" in methods_to_compare:
            print("Loading MAPPO results...")
            mappo_results_dict = {}
            for k in eval_K:
                mappo_file = get_marl_config_string("MAPPO", k, c_pos, c_phi, directed_transmission, jammer_on, testing=testing)
                eval_data_file_path = os.path.join(marl_eval_folder, "MAPPO", f"{mappo_file}.csv")
                if os.path.exists(eval_data_file_path):
                    data = pd.read_csv(eval_data_file_path)
                    mappo_results_dict[k] = data.to_dict(orient='records')
                    print(f"  Loaded MAPPO K={k}: {len(mappo_results_dict[k])} entries")
                else:
                    print(f"  Warning: MAPPO file not found for K={k}: {eval_data_file_path}")
            results_dicts["MAPPO"] = mappo_results_dict
        
        # Filter all results using value, delivery_time, and agent_sum_distance thresholds
        print(f"\nFiltering results with value [{value_remove_below}, {value_remove_above}], "
            f"delivery_time [{time_remove_below}, {time_remove_above}], "
            f"agent_sum_distance [{dist_remove_below}, {dist_remove_above}]...")
        
        for k in eval_K:
            # Find valid indices for all loaded methods
            valid_indices_dict = {}
            for method_name, method_dict in results_dicts.items():
                if k in method_dict:
                    valid_indices_dict[method_name] = {i for i, r in enumerate(method_dict[k]) 
                                                    if (value_remove_below <= r.get('value', float('-inf')) <= value_remove_above and
                                                        time_remove_below <= r.get('delivery_time', float('-inf')) <= time_remove_above and
                                                        dist_remove_below <= r.get('agent_sum_distance', float('-inf')) <= dist_remove_above)}
            
            # Keep only indices that are valid in ALL methods
            if valid_indices_dict:
                common_indices = set.intersection(*valid_indices_dict.values())
                
                # Filter all methods using the common indices
                for method_name, method_dict in results_dicts.items():
                    if k in method_dict:
                        num_before = len(method_dict[k])
                        method_dict[k] = [r for i, r in enumerate(method_dict[k]) if i in common_indices]
                        print(f"  {method_name} K={k}: {len(method_dict[k])} entries after filtering (removed {num_before - len(method_dict[k])})")
        
        # Create pairwise comparisons from selected methods
        from itertools import combinations
        method_list = sorted(results_dicts.keys())
        comparisons = []
        for method1, method2 in combinations(method_list, 2):
            comparisons.append((method1, method2, results_dicts[method1], results_dicts[method2]))
        
        if comparisons:
            eval_plot_dir = os.path.join(marl_eval_folder, "Compare", "All_heatmaps")
            os.makedirs(eval_plot_dir, exist_ok=True)
            
            # Create a descriptive filename with methods and scenario
            k_string = "K" + "_".join(map(str, eval_K))
            methods_str = "_vs_".join(sorted(results_dicts.keys()))
            scenario_str = f"{k_string}_dir{int(directed_transmission)}_jam{int(jammer_on)}"
            
            # Pass this info to the plotting function so it can use it in the saved filename
            plot_comparison_heatmap_all(comparisons, eval_K, plot_dir=eval_plot_dir, 
                                    methods_str=methods_str, scenario_str=scenario_str)
        else:
            print("Error: Need at least 2 methods to compare!")
    
    elif eval_mode == 20:  # Compare baseline values for two various K
        """
        Compare value distributions between multiple K values.
        Plots histograms vertically with separate axis limits for each K using seaborn.
        """
        K_compare_20 = [2, 10, 20]

        sns.set_style("whitegrid")
        sns.set_palette("husl")

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern']
        plt.rcParams['axes.labelsize'] = 25
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['legend.fontsize'] = 30
        plt.rcParams['figure.titlesize'] = 14
        
        
        # Load evaluation results for all K values
        results_dict = {}
        for k in K_compare_20:
            eval_data_file_path = os.path.join(result_dir, conf_string, f"evaluation_results_K{k}_n{row_end}_{conf_string}.csv")
            
            if not os.path.exists(eval_data_file_path):
                print(f"Warning: File not found: {eval_data_file_path}")
                continue
            
            data = pd.read_csv(eval_data_file_path)
            results_dict[k] = data.to_dict(orient='records')
            print(f"Loaded K={k}: {len(results_dict[k])} evaluations")
        
        # Create figure with len(K_compare_20) subplots vertically
        num_k = len(results_dict)
        fig, axes = plt.subplots(num_k, 1, figsize=(12, 3*num_k))
        
        # Handle case where there's only one K value
        if num_k == 1:
            axes = [axes]
        
        # Plot histograms for each K value using seaborn
        for idx, k in enumerate(sorted(results_dict.keys())):
            ax = axes[idx]
            values = [r['value'] for r in results_dict[k] if 'value' in r and not np.isnan(r['value'])]
            
            if values:
                # Create DataFrame for seaborn
                df_plot = pd.DataFrame({'value': values})
                
                # Use seaborn histplot with KDE
                sns.histplot(data=df_plot, x='value', stat='density', bins=50, 
                            ax=ax, color='skyblue', edgecolor='black', linewidth=1.5)
                
                # Add statistics
                mean_val = np.mean(values)
                median_val = np.median(values)
                std_val = np.std(values)
                
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend([f'$K$={k}'], loc='upper left', fontsize=25)

                # Only add x-label for the bottom subplot
                ax.set_ylabel('density')
                if idx == num_k - 1:
                    ax.set_xlabel('value')
                else:
                    ax.set_xlabel('')
                
                # Print summary statistics
                print(f"\nK={k} Statistics:")
                print(f"  Mean: {mean_val:.4f}")
                print(f"  Median: {median_val:.4f}")
                print(f"  Std Dev: {std_val:.4f}")
                print(f"  Min: {np.min(values):.4f}")
                print(f"  Max: {np.max(values):.4f}")
            else:
                ax.text(0.5, 0.5, f'No data for K={k}', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        
        # Overall title
        #fig.suptitle(fr"Value histograms over $K$", fontsize=14, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # Save plot
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            k_str = "_".join(str(k) for k in sorted(results_dict.keys()))
            plot_path = os.path.join(plot_dir, f'comparison_K{k_str}_values.pdf')
            plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
            print(f"Saved comparison plot to {plot_path}")
        else:
            plt.show()
    
    elif eval_mode == 21:  # 2x2 plot comparing trajectories between all scenarios
        """
        Create a 2x2 subplot figure comparing baseline trajectories across all scenarios:
        - (0,0): directed=False, jammer=False
        - (0,1): directed=True, jammer=False
        - (1,0): directed=False, jammer=True
        - (1,1): directed=True, jammer=True
        """
        k = 5
        row = 1
        method = "Baseline"
        force_gen_traj = True
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        if testing:
            traj_dir = f"Testing/Data/Trajectories/{method}"
            data_dir = "Testing/Data/Test_states"
            plot_dir = f"Testing/Plots/{method}/Trajectories/Comparison_all_scenarios"
        else:
            traj_dir = f"Evaluation/Trajectories/{method}"
            data_dir = "Evaluation/Evaluation_states/Data"
            plot_dir = f"Evaluation/Plots/{method}/Trajectories/Comparison_all_scenarios"
        
        # Define all scenario combinations
        scenarios = [
            (False, False, "dir0_jam0"),
            (True, False, "dir1_jam0"),
            (False, True, "dir0_jam1"),
            (True, True, "dir1_jam1")
        ]
        
        # Create 2x2 figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        axes = axes.flatten()
        
        # Set up LaTeX rendering and seaborn style
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern']
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.titlesize'] = 14
        
        
        Rcom = 1.0
        agent_color = 'orange'
        passive_color = 'black'
        
        # First pass: collect all axis limits across all scenarios
        all_x_coords = []
        all_y_coords = []
        
        for directed_transmission, jammer_on, scenario_name in scenarios:
            traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
            traj_file_path = os.path.join(traj_dir, traj_filename)
            
            if not os.path.exists(traj_file_path) or force_gen_traj:
                print(f"Generating trajectory for scenario {scenario_name}...")
                generate_baseline_trajectory(k, row, data_dir, c_pos, c_phi, 
                                            directed_transmission=directed_transmission, 
                                            jammer_on=jammer_on, 
                                            testing=testing, no_metrics=False)
            
            traj_data = pd.read_csv(traj_file_path)
            
            # Collect all coordinates
            for agent_id in range(1, k+1):
                agent_col_x = f'agent{agent_id}_x'
                agent_col_y = f'agent{agent_id}_y'
                if agent_col_x in traj_data.columns:
                    all_x_coords.extend(traj_data[agent_col_x].dropna().values)
                    all_y_coords.extend(traj_data[agent_col_y].dropna().values)
            
            # Collect jammer coordinates if present
            if 'jammer_x' in traj_data.columns:
                all_x_coords.extend(traj_data['jammer_x'].dropna().values)
                all_y_coords.extend(traj_data['jammer_y'].dropna().values)
            
            R = traj_data['R'].iloc[0] if 'R' in traj_data.columns else 10.0
            all_x_coords.extend([0, R])
            all_y_coords.extend([0, 0])
        
        # Compute common axis limits
        if all_x_coords and all_y_coords:
            R = 10.0
            Ra = 0.6 * R
            cap_radius = 1.5 * Rcom
            cap_height = 3 * Rcom  # Height of jammer capsule
            
            x_min = -cap_radius - 0.1 * Rcom
            x_max = R + cap_radius + 0.1 * Rcom
            y_min = -cap_height/2 - 0.5 * Rcom  # Just outside capsule bottom
            y_max = cap_height/2 + 0.5 * Rcom   # Just outside capsule top
        else:
            x_min, x_max = -2, 12
            y_min, y_max = -6, 6
        
        # Second pass: plot all scenarios with common axis limits
        for subplot_idx, (directed_transmission, jammer_on, scenario_name) in enumerate(scenarios):
            ax = axes[subplot_idx]
            
            traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
            traj_file_path = os.path.join(traj_dir, traj_filename)
            
            traj_data = pd.read_csv(traj_file_path)
            
            # Get list of active agents
            active_agents = []
            for agent_id in range(1, k+1):
                agent_col_has_message = f'agent{agent_id}_has_message'
                if np.any(traj_data[agent_col_has_message]):
                    active_agents.append(agent_id)
            
            R = traj_data['R'].iloc[0] if 'R' in traj_data.columns else 10.0
            Ra = 0.6 * R
            
            # Plot Rcom circle around transmitter
            if jammer_on:
                t_retrieve = len(traj_data) - 1
                for agent_k in range(1, k+1):
                    agent_col_has_message = f'agent{agent_k}_has_message'
                    times_with_message = traj_data.index[traj_data[agent_col_has_message] == True].tolist()
                    if times_with_message:
                        t_retrieve = min(t_retrieve, times_with_message[0])
                t_retrieve = max(0, t_retrieve - 1)
                
                jammer_x = traj_data['jammer_x'].iloc[t_retrieve] if 'jammer_x' in traj_data.columns else R/2
                jammer_y = traj_data['jammer_y'].iloc[t_retrieve] if 'jammer_y' in traj_data.columns else 0.0
                p_jammer = np.array([jammer_x, jammer_y])
                
                Rcom_tx = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=np.array([0, 0]))
            else:
                Rcom_tx = Rcom
            
            tx_circle = patches.Circle((0, 0), radius=Rcom_tx, color='blue', 
                                    fill=False, linestyle='--', linewidth=1.5, alpha=0.5)
            ax.add_patch(tx_circle)
            
            # Plot agent init area
            init_circle = patches.Circle((R/2, 0), radius=Ra, color='gray', 
                                        fill=False, linestyle=':', linewidth=1.5, alpha=0.4)
            ax.add_patch(init_circle)
            
            # Plot jammer capsule if active
            if jammer_on:
                cap_height = 3 * Rcom
                cap_radius = 1.5 * Rcom
                ax.plot([0, R], [cap_height/2, cap_height/2], color='red', lw=2, alpha=0.4)
                ax.plot([0, R], [-cap_height/2, -cap_height/2], color='red', lw=2, alpha=0.4)
                theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
                ax.plot(cap_radius * np.cos(theta_left), cap_radius * np.sin(theta_left), 
                    color='red', lw=2, alpha=0.4)
                theta_right = np.linspace(-np.pi/2, np.pi/2, 100)
                ax.plot(R + cap_radius * np.cos(theta_right), cap_radius * np.sin(theta_right), 
                    color='red', lw=2, alpha=0.4)
            
            # Plot all agent trajectories and their communication radii
            agent_label_added = False
            passive_agent_label_added = False
            
            for agent_id in range(1, k+1):
                agent_col_x = f'agent{agent_id}_x'
                agent_col_y = f'agent{agent_id}_y'
                agent_col_phi = f'agent{agent_id}_phi'
                
                if agent_col_x not in traj_data.columns:
                    continue
                
                initial_x = traj_data[agent_col_x].iloc[0]
                initial_y = traj_data[agent_col_y].iloc[0]
                final_x = traj_data[agent_col_x].iloc[-1]
                final_y = traj_data[agent_col_y].iloc[-1]
                
                is_passive = agent_id not in active_agents
                
                if is_passive:
                    ax.scatter(initial_x, initial_y, s=100, alpha=0.8, edgecolors='darkgrey', 
                            linewidth=1.5, color=passive_color, marker='o', zorder=4)
                    
                    if not passive_agent_label_added:
                        ax.plot([], [], alpha=0.8, linewidth=1.5, color=passive_color, marker='o', 
                            markersize=6, linestyle='none', label='Passive')
                        passive_agent_label_added = True
                else:
                    ax.scatter(initial_x, initial_y, s=80, alpha=0.5, edgecolors='darkgrey', 
                            linewidth=1, color='grey', marker='o', zorder=3)
                    
                    # Plot trajectory segments with fading
                    x_coords = traj_data[agent_col_x].values
                    y_coords = traj_data[agent_col_y].values
                    t_max = len(x_coords) - 1
                    
                    for t in range(len(x_coords) - 1):
                        alpha_val = 0.2 + 0.8 * (t / max(t_max, 1))
                        ax.plot(x_coords[t:t+2], y_coords[t:t+2], 
                            alpha=alpha_val, linewidth=1.5, color=agent_color, zorder=0)
                    
                    # Final position
                    ax.scatter(final_x, final_y, s=100, alpha=0.8, edgecolors='darkorange', 
                            linewidth=1.5, color=agent_color, marker='o', zorder=11)
                    
                    # Get position of jammer for when the current agent stops moving
                    # Compute time and position at which current agent stops moving
                    if jammer_on:
                        t_stop = len(traj_data) - 1
                        for t in range(len(traj_data)-1, 0, -1):
                            if (traj_data[agent_col_x].iloc[t] != traj_data[agent_col_x].iloc[t-1] or
                                traj_data[agent_col_y].iloc[t] != traj_data[agent_col_y].iloc[t-1]):
                                t_stop = t
                                break
                        p_agent_k = np.array([final_x, final_y])

                        # Compute jammer position at relay
                        jammer_x = traj_data['jammer_x'].iloc[t_stop] if 'jammer_x' in traj_data.columns else R/2
                        jammer_y = traj_data['jammer_y'].iloc[t_stop] if 'jammer_y' in traj_data.columns else 0.0
                        p_jammer = np.array([jammer_x, jammer_y])
                    else:
                        p_jammer = None
                    
                    # Plot communication radius at final position
                    if jammer_on:
                        Rcom_k = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=np.array([final_x, final_y]))
                    else:
                        Rcom_k = Rcom

                    # Only plot Rcom circle for isotropic (non-directed) transmission
                    if not directed_transmission:
                        agent_rcom_circle = patches.Circle((final_x, final_y), radius=Rcom_k, 
                                                        color=agent_color, fill=False, linestyle='--', 
                                                        linewidth=1.5, alpha=0.4, zorder=2)
                        ax.add_patch(agent_rcom_circle)
                    
                    # Plot antenna lobe at final position if directed transmission
                    if directed_transmission and agent_col_phi in traj_data.columns:
                        final_phi = traj_data[agent_col_phi].iloc[-1]
                        if jammer_on:
                            plot_antenna_lobe(ax, np.array([final_x, final_y]), final_phi, Rcom, 
                                            jammer_pos=p_jammer, color=agent_color)
                        else:
                            plot_antenna_lobe(ax, np.array([final_x, final_y]), final_phi, Rcom, color=agent_color)
                    
                    if not agent_label_added:
                        ax.plot([], [], alpha=0.6, linewidth=1.5, color=agent_color, label='Active')
                        agent_label_added = True
            
            # Plot transmitter and receiver
            ax.scatter(0, 0, s=250, marker='s', color='blue', alpha=0.8, 
                    edgecolors='darkblue', linewidth=1.5, zorder=5)
            ax.scatter(R, 0, s=250, marker='s', color='green', alpha=0.8, 
                    edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Plot jammer trajectory if present
            jammer_col_x = 'jammer_x'
            jammer_col_y = 'jammer_y'
            if jammer_col_x in traj_data.columns and jammer_col_y in traj_data.columns:
                jammer_data = traj_data[[jammer_col_x, jammer_col_y]].dropna()
                if len(jammer_data) > 0:
                    jammer_x_coords = jammer_data[jammer_col_x].values
                    jammer_y_coords = jammer_data[jammer_col_y].values
                    jammer_t_max = len(jammer_x_coords) - 1
                    
                    for t in range(len(jammer_x_coords) - 1):
                        alpha_val = 0.2 + 0.8 * (t / max(jammer_t_max, 1))
                        ax.plot(jammer_x_coords[t:t+2], jammer_y_coords[t:t+2], 
                            alpha=alpha_val, linewidth=1.5, linestyle='--', color='red', zorder=0)
                    
                    final_jammer_x = jammer_x_coords[-1]
                    final_jammer_y = jammer_y_coords[-1]
                    ax.scatter(final_jammer_x, final_jammer_y, s=150, marker='^', 
                            color='red', alpha=0.8, edgecolors='darkred', linewidth=1.5, zorder=10)
            
            # Set axis properties
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Use common axis limits for all subplots
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Add legend and subtitle
            #ax.legend(loc='upper left', fontsize=11)
            subtitle = f"directed={int(directed_transmission)}, jammer={int(jammer_on)}"
            #ax.set_title(subtitle, fontsize=12, fontweight='bold')
            
            ax.grid(False)
        
        # Overall title
        #fig.suptitle(f'{method} Trajectories: All Scenarios (K={k}, Row={row})', 
        #            fontsize=14, fontweight='bold')
        #plt.tight_layout(pad=-0.1, w_pad=-0.1, h_pad=-0.1)
        # plt.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, wspace=-0.1, hspace=-0.1) 
        
        # 2x2
        #plt.subplots_adjust(left=0.001, right=0.6, top=0.8, bottom=0.1, wspace=-0.05, hspace=-0.75)
        
        # 4x1
        plt.subplots_adjust(left=0.001, right=0.6, top=0.8, bottom=0.1, wspace=-0.05, hspace=0)
        

        # Save plot
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
            plot_filename = f"{method.lower()}_K{k}_row{row}_all_scenarios.pdf"
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
            print(f"Saved 2x2 comparison plot to {plot_path}")
        else:
            plt.show()
                                        
    elif eval_mode == 22:  # 2x2 animation comparing trajectories between all scenarios
        """
        Create a 2x2 animated figure comparing baseline trajectories across all scenarios:
        - (0,0): directed=False, jammer=False
        - (0,1): directed=True, jammer=False
        - (1,0): directed=False, jammer=True
        - (1,1): directed=True, jammer=True
        """
        k = 20
        row = 1
        method = "Baseline"
        force_gen_traj = True
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern']
        
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        if testing:
            traj_dir = f"Testing/Data/Trajectories/{method}"
            data_dir = "Testing/Data/Test_states"
            anim_dir = f"Testing/Animation/{method}/Trajectories/Comparison_all_scenarios"
        else:
            traj_dir = f"Evaluation/Trajectories/{method}"
            data_dir = "Evaluation/Evaluation_states/Data"
            anim_dir = f"Evaluation/Animation/{method}/Trajectories/Comparison_all_scenarios"
        
        # Define all scenario combinations
        scenarios = [
            (False, False, "dir0_jam0"),
            (True, False, "dir1_jam0"),
            (False, True, "dir0_jam1"),
            (True, True, "dir1_jam1")
        ]
        
        Rcom = 1.0
        agent_color_no_message = 'orange'
        agent_color_with_message = 'green'
        passive_color = 'black'
        
        # Load all trajectories and precompute agent statuses
        all_traj_data = {}
        all_active_agents = {}
        
        for directed_transmission, jammer_on, scenario_name in scenarios:
            traj_filename = f"{method.lower()}_K{k}_row{row}_dir{int(directed_transmission)}_jam{int(jammer_on)}_trajectory.csv"
            traj_file_path = os.path.join(traj_dir, traj_filename)
            
            if not os.path.exists(traj_file_path) or force_gen_traj:
                print(f"Generating trajectory for scenario {scenario_name}...")
                generate_baseline_trajectory(k, row, data_dir, c_pos, c_phi, 
                                            directed_transmission=directed_transmission, 
                                            jammer_on=jammer_on, 
                                            testing=testing, no_metrics=False)
            
            traj_data = pd.read_csv(traj_file_path)
            all_traj_data[(directed_transmission, jammer_on)] = traj_data
            
            # Get list of active agents
            active_agents = []
            for agent_id in range(1, k+1):
                agent_col_has_message = f'agent{agent_id}_has_message'
                if np.any(traj_data[agent_col_has_message]):
                    active_agents.append(agent_id)
            all_active_agents[(directed_transmission, jammer_on)] = active_agents
        
        # Determine common trajectory length
        num_frames = max([len(traj_data) for traj_data in all_traj_data.values()])
        
        # Extract R and Ra from any trajectory
        R = all_traj_data[(False, False)]['R'].iloc[0] if 'R' in all_traj_data[(False, False)].columns else 10.0
        Ra = 0.6 * R
        
        # Compute common axis limits
        cap_radius = 1.5 * Rcom
        cap_height = 3 * Rcom
        x_min = -cap_radius - 0.1 * Rcom
        x_max = R + cap_radius + 0.1 * Rcom
        #y_min = -cap_height/2 - 0.5 * Rcom
        #y_max = cap_height/2 + 0.5 * Rcom
        y_min = -Ra
        y_max = Ra
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        # Precompute static elements for each subplot
        for subplot_idx, (directed_transmission, jammer_on, scenario_name) in enumerate(scenarios):
            ax = axes[subplot_idx]
            
            traj_data = all_traj_data[(directed_transmission, jammer_on)]
            active_agents = all_active_agents[(directed_transmission, jammer_on)]
            
            # Plot Rcom circle around transmitter (static at first frame)
            if jammer_on:
                t_retrieve = len(traj_data) - 1
                for agent_k in range(1, k+1):
                    agent_col_has_message = f'agent{agent_k}_has_message'
                    times_with_message = traj_data.index[traj_data[agent_col_has_message] == True].tolist()
                    if times_with_message:
                        t_retrieve = min(t_retrieve, times_with_message[0])
                t_retrieve = max(0, t_retrieve - 1)
                
                jammer_x = traj_data['jammer_x'].iloc[t_retrieve] if 'jammer_x' in traj_data.columns else R/2
                jammer_y = traj_data['jammer_y'].iloc[t_retrieve] if 'jammer_y' in traj_data.columns else 0.0
                p_jammer = np.array([jammer_x, jammer_y])
                
                Rcom_tx = communication_range(theta=0, phi=0, C_dir=0.0, SINR_threshold=1.0, 
                                            jammer_pos=p_jammer, p_tx=np.array([0, 0]))
            else:
                Rcom_tx = Rcom
            
            tx_circle = patches.Circle((0, 0), radius=Rcom_tx, color='blue', 
                                    fill=False, linestyle='--', linewidth=1.5, alpha=0.5)
            ax.add_patch(tx_circle)
            
            # Plot agent init area
            init_circle = patches.Circle((R/2, 0), radius=Ra, color='gray', 
                                        fill=False, linestyle=':', linewidth=1.5, alpha=0.4)
            ax.add_patch(init_circle)
            
            # Plot jammer capsule if active
            if jammer_on:
                ax.plot([0, R], [cap_height/2, cap_height/2], color='red', lw=2, alpha=0.4)
                ax.plot([0, R], [-cap_height/2, -cap_height/2], color='red', lw=2, alpha=0.4)
                theta_left = np.linspace(np.pi/2, 3*np.pi/2, 100)
                ax.plot(cap_radius * np.cos(theta_left), cap_radius * np.sin(theta_left), 
                    color='red', lw=2, alpha=0.4)
                theta_right = np.linspace(-np.pi/2, np.pi/2, 100)
                ax.plot(R + cap_radius * np.cos(theta_right), cap_radius * np.sin(theta_right), 
                    color='red', lw=2, alpha=0.4)
            
            # Plot transmitter and receiver
            ax.scatter(0, 0, s=250, marker='s', color='blue', alpha=0.8, 
                    edgecolors='darkblue', linewidth=1.5, zorder=5)
            ax.scatter(R, 0, s=250, marker='s', color='green', alpha=0.8, 
                    edgecolors='darkgreen', linewidth=1.5, zorder=5)
            
            # Set axis properties
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(False)
        
        # Track static elements count
        num_static_patches = [len(ax.patches) for ax in axes]
        num_static_lines = [len(ax.get_lines()) for ax in axes]
        num_static_collections = [len(ax.collections) for ax in axes]
        
        # Generate animation frames
        pil_frames = []
        dur_frame = 100
        
        print(f"Generating {num_frames} animation frames for 2x2 comparison...")
        
        for t in range(num_frames):
            # Clear dynamic content for each subplot
            for subplot_idx, ax in enumerate(axes):
                while len(ax.get_lines()) > num_static_lines[subplot_idx]:
                    ax.get_lines()[-1].remove()
                while len(ax.patches) > num_static_patches[subplot_idx]:
                    ax.patches[-1].remove()
                while len(ax.collections) > num_static_collections[subplot_idx]:
                    ax.collections[-1].remove()
            
            # Plot for each scenario
            for subplot_idx, (directed_transmission, jammer_on, scenario_name) in enumerate(scenarios):
                ax = axes[subplot_idx]
                traj_data = all_traj_data[(directed_transmission, jammer_on)]
                active_agents = all_active_agents[(directed_transmission, jammer_on)]
                
                # Determine which time frame to plot (hold final frame if trajectory finished)
                plot_t = min(t, len(traj_data) - 1)
                
                # Skip if we're beyond the trajectory length and already plotted final frame
                if t >= len(traj_data) and plot_t == len(traj_data) - 1:
                    # Trajectory finished, keep plotting the final frame
                    pass
                
                # Get jammer info
                p_jammer = None
                if jammer_on and 'jammer_x' in traj_data.columns:
                    jammer_data = traj_data[['jammer_x', 'jammer_y']].iloc[:plot_t+1].dropna()
                    if len(jammer_data) > 0:
                        final_jammer_x = jammer_data['jammer_x'].iloc[-1]
                        final_jammer_y = jammer_data['jammer_y'].iloc[-1]
                        p_jammer = np.array([final_jammer_x, final_jammer_y])
                        
                        # Plot jammer trajectory
                        ax.plot(jammer_data['jammer_x'], jammer_data['jammer_y'], 
                            alpha=0.6, linewidth=2, linestyle='--', color='red', zorder=0)
                        
                        # Plot jammer position
                        ax.scatter(final_jammer_x, final_jammer_y, s=150, marker='^', 
                                color='red', alpha=0.8, edgecolors='darkred', linewidth=1.5, zorder=6)
                
                # Plot agent trajectories
                agent_label_added = False
                passive_agent_label_added = False
                
                for agent_id in range(1, k+1):
                    agent_col_x = f'agent{agent_id}_x'
                    agent_col_y = f'agent{agent_id}_y'
                    agent_col_phi = f'agent{agent_id}_phi'
                    agent_col_has_message = f'agent{agent_id}_has_message'
                    
                    if agent_col_x not in traj_data.columns:
                        continue
                    
                    initial_x = traj_data[agent_col_x].iloc[0]
                    initial_y = traj_data[agent_col_y].iloc[0]
                    
                    is_passive = agent_id not in active_agents
                    
                    if is_passive:
                        ax.scatter(initial_x, initial_y, s=100, alpha=0.8, edgecolors='darkgrey', 
                                linewidth=1.5, color=passive_color, marker='o', zorder=4)
                        
                        if not passive_agent_label_added:
                            ax.plot([], [], alpha=0.8, linewidth=1.5, color=passive_color, marker='o', 
                                markersize=6, linestyle='none', label='Passive')
                            passive_agent_label_added = True
                    else:
                        x_coords = traj_data[agent_col_x].values[:plot_t+1]
                        y_coords = traj_data[agent_col_y].values[:plot_t+1]
                        
                        if len(x_coords) > 0:
                            # Check if agent has message at current time
                            has_message_current = False
                            if agent_col_has_message in traj_data.columns:
                                has_message_current = bool(traj_data[agent_col_has_message].iloc[plot_t])
                            
                            # Determine agent color based on message status
                            agent_color = agent_color_with_message if has_message_current else agent_color_no_message
                            
                            # Plot initial position
                            if t == 0:
                                ax.scatter(initial_x, initial_y, s=80, alpha=0.5, edgecolors='darkgrey', 
                                        linewidth=1, color='grey', marker='o', zorder=3)
                            
                            # Plot trajectory segments
                            for time_idx in range(len(x_coords) - 1):
                                alpha_val = 0.2 + 0.8 * (time_idx / max(len(x_coords) - 1, 1))
                                ax.plot(x_coords[time_idx:time_idx+2], y_coords[time_idx:time_idx+2], 
                                    alpha=alpha_val, linewidth=1.5, color=agent_color, zorder=0)
                            
                            # Current position
                            current_x = x_coords[-1]
                            current_y = y_coords[-1]
                            
                            ax.scatter(current_x, current_y, s=100, alpha=0.8, 
                                    edgecolors=agent_color, linewidth=1.5, color=agent_color, 
                                    marker='o', zorder=11)
                            
                            # Plot Rcom circle for isotropic transmission only
                            if not directed_transmission:
                                if jammer_on and p_jammer is not None:
                                    Rcom_k = communication_range(theta=0, phi=0, C_dir=0.0, 
                                                        SINR_threshold=1.0, jammer_pos=p_jammer, 
                                                        p_tx=np.array([current_x, current_y]))
                                else:
                                    Rcom_k = Rcom
                                
                                agent_rcom = patches.Circle((current_x, current_y), radius=Rcom_k, 
                                                        color=agent_color, fill=False, linestyle='--', 
                                                        linewidth=1.5, alpha=0.4, zorder=2)
                                ax.add_patch(agent_rcom)
                            
                            # Plot antenna lobe for directed transmission
                            if directed_transmission and agent_col_phi in traj_data.columns:
                                current_phi = traj_data[agent_col_phi].iloc[plot_t]
                                plot_antenna_lobe(ax, np.array([current_x, current_y]), current_phi, Rcom, 
                                                jammer_pos=p_jammer, color=agent_color)
                            
                            if not agent_label_added:
                                ax.plot([], [], alpha=0.6, linewidth=1.5, color=agent_color_no_message, label='Active (no msg)')
                                ax.plot([], [], alpha=0.6, linewidth=1.5, color=agent_color_with_message, label='Active (with msg)')
                                agent_label_added = True
                
                # Add legend and subtitle
                if t == 0:  # Only add once
                    ax.legend(loc='upper left', fontsize=11)
                subtitle = f"directed={int(directed_transmission)}, jammer={int(jammer_on)}"
                ax.set_title(subtitle, fontsize=12, fontweight='bold')
            
            # Update overall title with frame number
            fig.suptitle(f'{method} Trajectories: All Scenarios (K={k}, Row={row}) - Frame {t}/{num_frames}', 
                        fontsize=14, fontweight='bold')
            
            # Draw and capture
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image_array = np.frombuffer(buf, dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image_rgb = image_array[:, :, :3]
            
            pil_frames.append(Image.fromarray(image_rgb, 'RGB'))
            
            if (t + 1) % max(1, num_frames // 10) == 0 or t == 0:
                print(f"  Generated frame {t+1}/{num_frames}")
        
        plt.close(fig)
        
        # Save GIF
        if anim_dir:
            os.makedirs(anim_dir, exist_ok=True)
            gif_path = os.path.join(anim_dir, 
                                f"{method.lower()}_K{k}_row{row}_all_scenarios.gif")
            
            print(f"\nSaving animation to {gif_path}...")
            pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], 
                            duration=dur_frame, loop=0)
            
            print(f"✓ Saved animation to {gif_path}")
            print(f"  Total frames: {len(pil_frames)}")
            print(f"  Duration per frame: {dur_frame}ms")
        else:
            print("Animation frames generated (not saved)")

if __name__ == "__main__":
    main()