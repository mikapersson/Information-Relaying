import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

def antenna_gain(theta, C_dir=1.0):
    """Calculate antenna gain |a(0)^H * a(theta)|"""
    # Steering vectors
    a_0 = np.array([1, C_dir])  # a(0)
    a_theta = np.array([1, C_dir * np.exp(1j * np.pi * np.sin(theta))])  # a(theta)
    
    # Antenna gain: |a(0)^H * a(theta)|
    gain = np.abs(np.conj(a_0) @ a_theta)
    return gain

def communication_range(theta, phi, C_dir=1.0, SINR_threshold=1.0, jammer_pos=None, p_tx=None):
    """
    Calculate communication range based on SINR at that angle.
    
    Args:
        theta: Angle relative to antenna boresight
        phi: Antenna orientation angle
        C_dir: Directivity parameter for the steering vector
        SINR_threshold: SINR threshold value
        jammer_pos: Jammer position [x, y], if None, no jammer
        p_tx: Transmitter position, defaults to origin
    
    Returns:
        Communication range R for the given theta
    """
    if p_tx is None:
        p_tx = np.array([0, 0])
    
    # For directed transmission (C_dir > 0), check if theta is within [-pi/2, pi/2]
    # For isotropic transmission (C_dir = 0), allow full range [-pi, pi]
    if C_dir > 0 and abs(theta) > np.pi/2:
        return 0.0
    
    # Calculate antenna gain |a(0)^H * a(theta)|
    antenna_gain_value = antenna_gain(theta, C_dir)
    
    # Set jammer coefficient
    C_jam = 3.0 if jammer_pos is not None else 0.0
    
    # For communication range calculation, we need to solve for R such that SINR = SINR_threshold = 1
    # SINR = antenna_gain / (R^2 * (1 + C_jam * ||p_r - p_j||^(-2)))
    # Where p_r = p_tx + R * [cos(theta + phi), sin(theta + phi)]
    
    if C_jam == 0:
        # No jammer case: SINR = antenna_gain / R^2
        # R = sqrt(antenna_gain / SINR_threshold)
        return np.sqrt(antenna_gain_value / SINR_threshold)
    else:
        # With jammer: need to solve iteratively
        actual_angle = theta + phi
        
        # Try different ranges to find where SINR = SINR_threshold
        R_values = np.linspace(0.01, 2.0, 1000)  # resolution of lobe
        best_R = 0.1
        
        for R in R_values:
            p_r = p_tx + R * np.array([np.cos(actual_angle), np.sin(actual_angle)])
            
            # Calculate SINR at this position
            dist_tr_squared = R**2  # ||p_r - p_t||^2
            dist_jr_squared = np.linalg.norm(p_r - jammer_pos)**2  # ||p_r - p_j||^2
            
            # SINR calculation according to equation
            denominator = dist_tr_squared * (1 + C_jam / dist_jr_squared)
            sinr = antenna_gain_value / denominator
            
            if sinr >= SINR_threshold:
                best_R = R
            else:
                break
        
        return best_R

def calculate_sinr_at_receiver(phi, C_dir=1.0, jammer_pos=None):
    """Calculate SINR at receiver position for given phi using the provided equation"""
    p_tx = np.array([0, 0])  # Transmitter at origin
    p_rec_angle = np.pi/4  # Receiver at angle pi/4
    p_rec_distance = np.sqrt(2)
    p_rec = np.array([p_rec_distance * np.cos(p_rec_angle), p_rec_distance * np.sin(p_rec_angle)])
    
    # Calculate theta_rec relative to antenna boresight
    theta_rec = np.mod(p_rec_angle - phi + np.pi, 2*np.pi) - np.pi
    
    # Check if theta_rec is within [-pi/2, pi/2]
    if abs(theta_rec) > np.pi/2:
        return 0.0, theta_rec
    
    # Calculate antenna gain using shared function
    antenna_gain_value = antenna_gain(theta_rec, C_dir)
    
    # Set jammer coefficient
    C_jam = 3.0 if jammer_pos is not None else 0.0
    
    # Calculate distances
    dist_tr_squared = np.linalg.norm(p_rec - p_tx)**2  # ||p_r - p_t||^2
    
    if C_jam > 0:
        dist_jr_squared = np.linalg.norm(p_rec - jammer_pos)**2  # ||p_r - p_j||^2
    else:
        dist_jr_squared = 1.0  # Dummy value when no jammer
    
    # SINR calculation according to equation
    if C_jam > 0:
        denominator = dist_tr_squared * (1 + C_jam / dist_jr_squared)
    else:
        denominator = dist_tr_squared  # When C_jam = 0, the (1 + C_jam * ...) term becomes 1
    
    sinr = antenna_gain_value / denominator
    
    return sinr, theta_rec

def calculate_max_range(phi, C_dir=1.0, SINR_threshold=1.0, jammer_pos=None):
    """Calculate maximum communication range for given phi"""
    p_tx = np.array([0, 0])
    
    # For directed transmission, use [-pi/2, pi/2]; for isotropic, use [-pi, pi]
    if C_dir > 0:
        theta_range = np.linspace(-np.pi/2, np.pi/2, 100)
    else:
        theta_range = np.linspace(-np.pi, np.pi, 200)  # More points for full circle
    
    ranges = [communication_range(theta, phi, C_dir, SINR_threshold, jammer_pos, p_tx) for theta in theta_range]
    return max(ranges) if ranges else 0.0

def get_default_jammer_position():
    """Get default jammer position (1 unit below receiver)"""
    p_rec_distance = np.sqrt(2)
    p_rec_angle = np.pi/4
    p_rec = np.array([p_rec_distance * np.cos(p_rec_angle), p_rec_distance * np.sin(p_rec_angle)])
    return p_rec + np.array([0, -1])

def draw_common_elements(ax, p_tx, p_rec, jammer_pos, current_phi, ranges, show_jammer=True):
    """Draw common elements used in both static and animated plots"""
    # Plot transmitter
    ax.plot(p_tx[0], p_tx[1], 'go', markersize=10, label='Transmitter', zorder=5)
    
    # Draw local x and y axes through transmitter
    axis_length = 1.5
    ax.arrow(p_tx[0] - axis_length, p_tx[1], 2 * axis_length, 0, head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=4)
    ax.arrow(p_tx[0], p_tx[1] - axis_length, 0, 2 * axis_length, head_width=0.05, head_length=0.05, fc='black', ec='black', zorder=4)
    ax.text(axis_length + 0.1, 0, 'x', fontsize=12, ha='left')
    ax.text(0, axis_length + 0.1, 'y', fontsize=12, ha='center')
    
    # Plot receiving agent
    ax.plot(p_rec[0], p_rec[1], 'bo', markersize=10, label='Receiving Agent', zorder=5)
    
    # Draw line from transmitter to receiver
    ax.plot([p_tx[0], p_rec[0]], [p_tx[1], p_rec[1]], 'k--', alpha=0.7, zorder=3)
    
    # Plot Rcom=1 circle around receiver and jammer (if jammer exists)
    if show_jammer and jammer_pos is not None:
        circle_angles = np.linspace(0, 2*np.pi, 100)
        rcom_circle_x = p_rec[0] + 1.0 * np.cos(circle_angles)
        rcom_circle_y = p_rec[1] + 1.0 * np.sin(circle_angles)
        ax.plot(rcom_circle_x, rcom_circle_y, 'g:', linewidth=2, alpha=0.7, label='Rcom=1 around receiver', zorder=2)
        
        # Plot jammer
        ax.plot(jammer_pos[0], jammer_pos[1], 'rs', markersize=10, label='Jammer', zorder=5)
    
    # Draw antenna orientation (phi)
    max_range = max(ranges) if ranges else 1.0
    antenna_length = max_range
    antenna_end = p_tx + antenna_length * np.array([np.cos(current_phi), np.sin(current_phi)])
    ax.arrow(p_tx[0], p_tx[1], antenna_end[0] - p_tx[0], antenna_end[1] - p_tx[1], 
              head_width=0.08, head_length=0.08, fc='green', ec='green', linewidth=3, 
              label='Antenna boresight', zorder=4)
    
    return antenna_end

def draw_theta_rec_arc(ax, p_tx, p_rec, current_phi, show_arrow=True):
    """Draw theta_rec arc between antenna boresight and receiver direction"""
    angle_to_receiver = np.arctan2(p_rec[1] - p_tx[1], p_rec[0] - p_tx[0])
    theta_rec = np.mod(angle_to_receiver - current_phi + np.pi, 2*np.pi) - np.pi
    
    theta_start = current_phi
    theta_end = angle_to_receiver
    angle_diff = np.mod(theta_end - theta_start + np.pi, 2*np.pi) - np.pi
    
    if abs(angle_diff) > 0.01:  # Only draw arc if there's a meaningful angle
        if angle_diff >= 0:
            theta_rec_arc = np.linspace(theta_start, theta_start + angle_diff, 50)
        else:
            theta_rec_arc = np.linspace(theta_start, theta_start + angle_diff, 50)
        
        arc_radius_theta_rec = 0.6
        theta_rec_arc_x = p_tx[0] + arc_radius_theta_rec * np.cos(theta_rec_arc)
        theta_rec_arc_y = p_tx[1] + arc_radius_theta_rec * np.sin(theta_rec_arc)
        ax.plot(theta_rec_arc_x, theta_rec_arc_y, 'k-', linewidth=5, zorder=6)
        
        # Add arrow if requested
        if show_arrow:
            arc_start = -8
            arrow_start_theta_rec = np.array([theta_rec_arc_x[arc_start], theta_rec_arc_y[arc_start]])
            if angle_diff >= 0:
                arrow_dir_theta_rec = np.array([theta_rec_arc_x[arc_start]-theta_rec_arc_x[arc_start-1], theta_rec_arc_y[arc_start]-theta_rec_arc_y[arc_start-1]])
            else:
                arrow_dir_theta_rec = np.array([theta_rec_arc_x[arc_start]-theta_rec_arc_x[arc_start-1], theta_rec_arc_y[arc_start]-theta_rec_arc_y[arc_start-1]])
            ax.arrow(arrow_start_theta_rec[0], arrow_start_theta_rec[1], arrow_dir_theta_rec[0], arrow_dir_theta_rec[1], 
                      head_width=0.02, head_length=0.02, fc='black', ec='black', zorder=6)
    
    return theta_rec

def plot_communication_setup():
    # Setup
    p_tx = np.array([0, 0])  # Transmitter at origin
    p_rec = np.array([2, 1])  # Receiver position
    phi = np.pi/3  # Antenna orientation
    
    # Calculate theta
    angle_to_receiver = np.arctan2(p_rec[1] - p_tx[1], p_rec[0] - p_tx[0])
    theta = np.mod(angle_to_receiver - phi + np.pi, 2*np.pi) - np.pi
    
    plt.figure(figsize=(10, 8))
    
    # Plot transmitter and receiver
    plt.plot(p_tx[0], p_tx[1], 'ro', markersize=10, label='Transmitter')
    plt.plot(p_rec[0], p_rec[1], 'bs', markersize=10, label='Receiver')
    
    # Draw local x and y axes through transmitter
    axis_length = 1.5
    plt.arrow(p_tx[0] - axis_length, p_tx[1], 2 * axis_length, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    plt.arrow(p_tx[0], p_tx[1] - axis_length, 0, 2 * axis_length, head_width=0.05, head_length=0.05, fc='black', ec='black')
    plt.text(axis_length + 0.1, 0, 'x', fontsize=12, ha='left')
    plt.text(0, axis_length + 0.1, 'y', fontsize=12, ha='center')
    
    # Draw line from transmitter to receiver
    plt.plot([p_tx[0], p_rec[0]], [p_tx[1], p_rec[1]], 'k--', alpha=0.7)
    
    # Draw antenna orientation (phi)
    antenna_length = 1.0
    antenna_end = p_tx + antenna_length * np.array([np.cos(phi), np.sin(phi)])
    plt.arrow(p_tx[0], p_tx[1], antenna_end[0] - p_tx[0], antenna_end[1] - p_tx[1], 
              head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2)
    
    # Draw phi arc with arrow
    phi_arc = np.linspace(0, phi, 50)
    arc_radius = 0.3
    phi_arc_x = p_tx[0] + arc_radius * np.cos(phi_arc)
    phi_arc_y = p_tx[1] + arc_radius * np.sin(phi_arc)
    plt.plot(phi_arc_x, phi_arc_y, 'g-', linewidth=2)
    
    # Add arrow to phi arc
    mid_phi = phi / 2
    arrow_start = p_tx + arc_radius * np.array([np.cos(mid_phi), np.sin(mid_phi)])
    arrow_dir = np.array([-np.sin(mid_phi), np.cos(mid_phi)]) * 0.05  # Tangent direction
    plt.arrow(arrow_start[0], arrow_start[1], arrow_dir[0], arrow_dir[1], 
              head_width=0.02, head_length=0.02, fc='green', ec='green')
    
    plt.text(0.4, 0.2, r'$\phi = \pi/3$', fontsize=12, color='green')
    
    # Draw theta arc with arrow
    theta_start = phi
    theta_end = angle_to_receiver
    if theta < 0:
        theta_arc = np.linspace(theta_end, theta_start, 50)
        arrow_direction = -1  # Clockwise
    else:
        theta_arc = np.linspace(theta_start, theta_end, 50)
        arrow_direction = 1   # Counter-clockwise
    
    arc_radius_theta = 0.5
    theta_arc_x = p_tx[0] + arc_radius_theta * np.cos(theta_arc)
    theta_arc_y = p_tx[1] + arc_radius_theta * np.sin(theta_arc)
    plt.plot(theta_arc_x, theta_arc_y, 'r-', linewidth=2)
    
    # Add arrow to theta arc
    mid_theta = (theta_start + theta_end) / 2
    arrow_start_theta = p_tx + arc_radius_theta * np.array([np.cos(mid_theta), np.sin(mid_theta)])
    arrow_dir_theta = arrow_direction * np.array([-np.sin(mid_theta), np.cos(mid_theta)]) * 0.05
    plt.arrow(arrow_start_theta[0], arrow_start_theta[1], arrow_dir_theta[0], arrow_dir_theta[1], 
              head_width=0.02, head_length=0.02, fc='red', ec='red')
    
    plt.text(0.3, 0.6, rf'$\theta = {theta:.2f}$', fontsize=12, color='red')
    
    # Labels and annotations
    plt.text(p_tx[0] - 0.1, p_tx[1] - 0.15, r'$p_t$ (origin)', fontsize=12, ha='center')
    plt.text(p_rec[0] + 0.1, p_rec[1], rf'$p_r$ = [{p_rec[0]}, {p_rec[1]}]', fontsize=12, ha='left')
    plt.text(antenna_end[0] + 0.1, antenna_end[1], 'Antenna\nboresight', fontsize=10, ha='left', color='green')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Communication Setup: Transmitter, Receiver, and Angles')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Phi (antenna orientation): {phi:.3f} rad = {np.degrees(phi):.1f}°")
    print(f"Theta (angle between boresight and receiver): {theta:.3f} rad = {np.degrees(theta):.1f}°")

def animate_communication_setup(save_animation=False):
    # Setup
    p_tx = np.array([0, 0])  # Transmitter at origin
    phi = np.pi/3  # Antenna orientation (fixed)
    R_receiver = 1.0  # Distance of receiver from transmitter
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw local x and y axes through transmitter
    axis_length = 1.5
    ax.arrow(p_tx[0], p_tx[1], axis_length, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.arrow(p_tx[0], p_tx[1], 0, axis_length, head_width=0.05, head_length=0.05, fc='black', ec='black')
    ax.text(axis_length + 0.1, 0, 'x', fontsize=12, ha='left')
    ax.text(0, axis_length + 0.1, 'y', fontsize=12, ha='center')
    
    # Draw transmitter
    ax.plot(p_tx[0], p_tx[1], 'ro', markersize=10, label='Transmitter')
    
    # Draw receiver circular path
    circle_angles = np.linspace(0, 2*np.pi, 100)
    circle_x = p_tx[0] + R_receiver * np.cos(circle_angles)
    circle_y = p_tx[1] + R_receiver * np.sin(circle_angles)
    ax.plot(circle_x, circle_y, 'k:', alpha=0.3, label='Receiver path')
    
    # Draw antenna orientation (phi) - fixed
    antenna_length = 1.0
    antenna_end = p_tx + antenna_length * np.array([np.cos(phi), np.sin(phi)])
    ax.arrow(p_tx[0], p_tx[1], antenna_end[0] - p_tx[0], antenna_end[1] - p_tx[1], 
             head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2, label='Antenna boresight')
    
    # Draw phi arc - fixed
    phi_arc = np.linspace(0, phi, 50)
    arc_radius = 0.3
    phi_arc_x = p_tx[0] + arc_radius * np.cos(phi_arc)
    phi_arc_y = p_tx[1] + arc_radius * np.sin(phi_arc)
    ax.plot(phi_arc_x, phi_arc_y, 'g-', linewidth=2)
    ax.text(0.4, 0.2, r'$\phi = \pi/3$', fontsize=12, color='green')
    
    # Initialize dynamic elements
    receiver_point, = ax.plot([], [], 'bs', markersize=10, label='Receiver')
    line_to_receiver, = ax.plot([], [], 'k--', alpha=0.7, label='Line to receiver')
    theta_arc, = ax.plot([], [], 'r-', linewidth=2)
    theta_text = ax.text(-1.5, 1.5, '', fontsize=12, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Animation: Receiver Rotating Around Transmitter')
    ax.legend(loc='upper right')
    
    def animate(frame):
        # Calculate receiver position
        angle = frame * 2 * np.pi / 200  # 200 frames for full rotation
        p_rec = p_tx + R_receiver * np.array([np.cos(angle), np.sin(angle)])
        
        # Calculate theta properly
        angle_to_receiver = np.arctan2(p_rec[1] - p_tx[1], p_rec[0] - p_tx[0])
        theta = np.mod(angle_to_receiver - phi + np.pi, 2*np.pi) - np.pi
        
        # Update receiver position
        receiver_point.set_data([p_rec[0]], [p_rec[1]])
        
        # Update line to receiver
        line_to_receiver.set_data([p_tx[0], p_rec[0]], [p_tx[1], p_rec[1]])
        
        # Update theta arc - fixed logic
        theta_start = phi
        theta_end = angle_to_receiver
        
        # Create arc from antenna boresight to receiver direction
        # Always take the shorter path
        angle_diff = np.mod(theta_end - theta_start + np.pi, 2*np.pi) - np.pi
        
        if angle_diff >= 0:
            # Counter-clockwise
            theta_arc_angles = np.linspace(theta_start, theta_start + angle_diff, 50)
        else:
            # Clockwise  
            theta_arc_angles = np.linspace(theta_start, theta_start + angle_diff, 50)
        
        arc_radius_theta = 0.5
        theta_arc_x = p_tx[0] + arc_radius_theta * np.cos(theta_arc_angles)
        theta_arc_y = p_tx[1] + arc_radius_theta * np.sin(theta_arc_angles)
        theta_arc.set_data(theta_arc_x, theta_arc_y)
        
        # Update theta text with proper range
        theta_text.set_text(rf'$\theta = {theta:.2f}$ rad' + '\n' + rf'$\theta = {np.degrees(theta):.1f}°$')
        
        return receiver_point, line_to_receiver, theta_arc, theta_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True, repeat=True)
    
    # Save animation if requested
    if save_animation:
        animation_dir = "Animations/communication"
        os.makedirs(animation_dir, exist_ok=True)
        animation_name = "communication_angle"
        animation_path = os.path.join(animation_dir, f"{animation_name}.gif")
        anim.save(animation_path, writer='ffmpeg', fps=20, bitrate=1800)
        print(f"Animation saved to: {animation_path}")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def animate_antenna_gain(save_animation=False, C_dir=1.0):
    """
    Animate the antenna gain (numerator of SINR) as a function of theta.
    
    Args:
        save_animation (bool): Whether to save the animation
        C_dir (float): Directivity parameter for the steering vector
    """
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Setup for polar plot (left subplot)
    theta_range = np.linspace(0, 2*np.pi, 1000)
    gains = [antenna_gain(theta, C_dir) for theta in theta_range]
    
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    ax1.plot(theta_range, gains, 'b-', linewidth=2, label=f'Antenna Gain (C_dir={C_dir})')
    ax1.set_title(r'Antenna Gain Pattern: $|a(0)^H a(\theta)|$', pad=20)
    ax1.set_xlabel(r'$\theta$ (radians)')
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    
    # Setup for Cartesian plot (right subplot)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(theta_range, gains, 'b-', linewidth=2, label=f'Antenna Gain (C_dir={C_dir})')
    ax2.set_xlabel(r'$\theta$ (radians)')
    ax2.set_ylabel('Antenna Gain $|a(0)^H a(\\theta)|$')
    ax2.set_title(r'Antenna Gain: $|a(0)^H a(\theta)|$ vs $\theta$')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, max(gains) * 1.1)
    
    # Set x-axis ticks for better readability
    theta_ticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    theta_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    ax2.set_xticks(theta_ticks)
    ax2.set_xticklabels(theta_labels)
    
    # Add animated marker
    current_theta = 0
    gain_at_theta = antenna_gain(current_theta, C_dir)
    
    # Markers for current position
    polar_marker, = ax1.plot([current_theta], [gain_at_theta], 'ro', markersize=8, label='Current Position')
    cartesian_marker, = ax2.plot([current_theta], [gain_at_theta], 'ro', markersize=8, label='Current Position')
    
    # Add vertical line in Cartesian plot
    vertical_line = ax2.axvline(current_theta, color='red', linestyle='--', alpha=0.7)
    
    # Text for current values
    value_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                         verticalalignment='top')
    
    # Update legends
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax2.legend()
    
    def animate(frame):
        # Calculate current theta (0 to 2π over 200 frames)
        current_theta = frame * 2 * np.pi / 200
        gain_at_theta = antenna_gain(current_theta, C_dir)
        
        # Update markers
        polar_marker.set_data([current_theta], [gain_at_theta])
        cartesian_marker.set_data([current_theta], [gain_at_theta])
        
        # Update vertical line
        vertical_line.set_xdata([current_theta])
        
        # Update text
        value_text.set_text(f'θ = {current_theta:.2f} rad\n'
                           f'θ = {np.degrees(current_theta):.1f}°\n'
                           f'Gain = {gain_at_theta:.3f}')
        
        return polar_marker, cartesian_marker, vertical_line, value_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=True, repeat=True)
    
    # Save animation if requested
    if save_animation:
        animation_dir = "Animations/communication"
        os.makedirs(animation_dir, exist_ok=True)
        animation_name = "antenna_gain"
        animation_path = os.path.join(animation_dir, f"{animation_name}.gif")
        anim.save(animation_path, writer='ffmpeg', fps=20, bitrate=1800)
        print(f"Animation saved to: {animation_path}")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def plot_communication_range(save_plot=False, C_dir=1.0, SINR_threshold=1.0, phi=np.pi/3, jammer_pos=None):
    """
    Plot the communication range R = sqrt(antenna_gain / SINR_threshold) as a function of theta.
    
    Args:
        save_plot (bool): Whether to save the plot
        C_dir (float): Directivity parameter for the steering vector
        SINR_threshold (float): SINR threshold value
        phi (float): Antenna orientation angle
        jammer_pos (array): Jammer position [x, y], if None, no jammer
    """
    
    # Use default jammer position if not provided
    if jammer_pos is None:
        jammer_pos = get_default_jammer_position()
    
    # Setup
    p_tx = np.array([0, 0])  # Transmitter at origin
    
    # Calculate receiver position (standard position)
    p_rec_distance = np.sqrt(2)
    p_rec_angle = np.pi/4
    p_rec = p_tx + p_rec_distance * np.array([np.cos(p_rec_angle), np.sin(p_rec_angle)])
    
    # Create theta range relative to antenna boresight
    theta_range = np.linspace(-np.pi/2, np.pi/2, 1000)
    
    # Calculate communication range for each theta
    ranges = [communication_range(theta, phi, C_dir, SINR_threshold, jammer_pos) for theta in theta_range]
    
    # Convert theta to actual angles (relative to antenna orientation phi)
    actual_angles = theta_range + phi
    
    # Convert to Cartesian coordinates for the range bubble
    range_x = p_tx[0] + np.array(ranges) * np.cos(actual_angles)
    range_y = p_tx[1] + np.array(ranges) * np.sin(actual_angles)
    
    plt.figure(figsize=(12, 10))
    
    # Draw common elements
    antenna_end = draw_common_elements(plt.gca(), p_tx, p_rec, jammer_pos, phi, ranges, show_jammer=True)
    
    # Draw phi arc
    phi_arc = np.linspace(0, phi, 50)
    arc_radius = 0.3
    phi_arc_x = p_tx[0] + arc_radius * np.cos(phi_arc)
    phi_arc_y = p_tx[1] + arc_radius * np.sin(phi_arc)
    plt.plot(phi_arc_x, phi_arc_y, 'g-', linewidth=2, zorder=3)
    
    # Add arrow to phi arc
    mid_phi = phi / 2
    arrow_start = p_tx + arc_radius * np.array([np.cos(mid_phi), np.sin(mid_phi)])
    arrow_dir = np.array([-np.sin(mid_phi), np.cos(mid_phi)]) * 0.05  # Tangent direction
    plt.arrow(arrow_start[0], arrow_start[1], arrow_dir[0], arrow_dir[1], 
              head_width=0.02, head_length=0.02, fc='green', ec='green', zorder=3)
    
    # Plot the communication range bubble
    jammer_text = " with Jammer" if jammer_pos is not None else ""
    plt.fill(range_x, range_y, alpha=0.1, color='blue', label=f'Communication Range{jammer_text}\n(SINR={SINR_threshold}, $C_{{dir}}$={C_dir})', zorder=1)
    plt.plot(range_x, range_y, 'b-', linewidth=2, zorder=2)
    
    # Draw concentric circles
    circle1_angles = np.linspace(0, 2*np.pi, 100)

    # Inner circle with radius 1
    circle1_x = p_tx[0] + 1.0 * np.cos(circle1_angles)
    circle1_y = p_tx[1] + 1.0 * np.sin(circle1_angles)
    plt.plot(circle1_x, circle1_y, 'k--', linewidth=2, alpha=0.8, label='Radius = Rcom', zorder=2)

    # Outer circle with radius = max range
    max_range = max(ranges)
    circle2_x = p_tx[0] + max_range * np.cos(circle1_angles)
    circle2_y = p_tx[1] + max_range * np.sin(circle1_angles)
    plt.plot(circle2_x, circle2_y, 'b--', linewidth=2, alpha=0.8, label=fr'Radius = {max_range:.2f} (Max Range)', zorder=2)
    
    # Labels and annotations
    plt.text(antenna_end[0] + 0.1, antenna_end[1], 'Antenna\nboresight', fontsize=10, ha='left', color='green')
    
    # Add formula in text box
    if jammer_pos is not None:
        formula_text = r'$R(\theta) = \sqrt{\frac{|a(0)^H a(\theta)|}{SINR_0(1+C_{jam}\|p_{rec}-p_{jam}\|^{-2})}}$'
    else:
        formula_text = r'$R(\theta) = \sqrt{\frac{|a(0)^H a(\theta)|}{SINR_0}}$'
    
    plt.text(0.02, 0.98, formula_text, transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
             verticalalignment='top')

    # Add text annotation for the receiving agent
    plt.text(p_rec[0] + 0.1, p_rec[1] + 0.1, rf'$p_r$ at $\sqrt{{2}}$, $\pi/4$', fontsize=10, ha='left')
    
    # Draw theta_rec arc with arrow
    theta_rec = draw_theta_rec_arc(plt.gca(), p_tx, p_rec, phi, show_arrow=True)
    plt.text(0.7, 0.4, rf'$\theta_{{rec}} = {theta_rec:.2f}$', fontsize=12, color='red')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Set limits based on the range
    max_range = max(ranges)
    plt.xlim(-max_range*1.2, max_range*1.2)
    plt.ylim(-max_range*1.2, max_range*1.2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Update title based on jammer presence
    title_suffix = " with Jammer" if jammer_pos is not None else ""
    plt.title(f'Communication Range Pattern{title_suffix}: $R(\\theta) = \\sqrt{{|a(0)^H a(\\theta)|/SINR}}$\n$C_{{dir}}$ = {C_dir}, SINR threshold = {SINR_threshold}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_dir = "Plots/communication"
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = "communication_range_pattern_with_jammer" if jammer_pos is not None else "communication_range_pattern"
        plot_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    plt.show()

def plot_communication_range_comparison(save_plot=False, SINR_threshold=1.0, phi=np.pi/3, 
                                      receiver_range=np.sqrt(2), receiver_angle=np.pi/4,
                                      jammer_range_from_receiver=1.0, jammer_angle_from_receiver=3*np.pi/2,
                                      include_text_boxes=True, marker_size=12, symbol_fontsize=16, 
                                      linewidth=3, legend_fontsize=20):
    """
    Plot communication range R(theta) for all four cases in one figure:
    1. Isotropic transmission without jammer (C_dir=0, no jammer)
    2. Isotropic transmission with jammer (C_dir=0, with jammer)
    3. Directed transmission without jammer (C_dir=1, no jammer)
    4. Directed transmission with jammer (C_dir=1, with jammer)
    
    Args:
        save_plot (bool): Whether to save the plot
        SINR_threshold (float): SINR threshold value
        phi (float): Antenna orientation angle
        receiver_range (float): Distance from transmitter (origin) to receiver
        receiver_angle (float): Angle from transmitter to receiver (in radians)
        jammer_range_from_receiver (float): Distance from receiver to jammer
        jammer_angle_from_receiver (float): Angle from receiver to jammer (in radians, relative to receiver)
        include_text_boxes (bool): Whether to include all text boxes in the plot
        marker_size (float): Size of markers (transmitter, receiver, jammer)
        symbol_fontsize (float): Font size for symbols (φ, θ, p_t, p_r)
        linewidth (float): Width of lines and arrows
        legend_fontsize (float): Font size for legend text
    """
    
    # Setup
    p_tx = np.array([0, 0])  # Transmitter at origin
    
    # Calculate receiver position based on range and angle from transmitter
    p_rec = p_tx + receiver_range * np.array([np.cos(receiver_angle), np.sin(receiver_angle)])
    
    # Calculate jammer position based on range and angle from receiver
    jammer_pos = p_rec + jammer_range_from_receiver * np.array([np.cos(jammer_angle_from_receiver), 
                                                                np.sin(jammer_angle_from_receiver)])
    
    # Create theta ranges - full range for isotropic, limited range for directed
    theta_eps=0.05
    theta_range_iso = np.linspace(-np.pi, np.pi, 2000)  # Full range for isotropic
    theta_range_dir = np.linspace(-np.pi/2+theta_eps, np.pi/2-theta_eps, 1000)  # Limited range for directed
    
    # Calculate communication range for all four cases
    # Case 1: Isotropic without jammer (C_dir=0, no jammer) - FULL RANGE
    ranges_iso_no_jam = [communication_range(theta, phi, C_dir=0.0, SINR_threshold=SINR_threshold, jammer_pos=None) for theta in theta_range_iso]
    
    # Case 2: Isotropic with jammer (C_dir=0, with jammer) - FULL RANGE
    ranges_iso_with_jam = [communication_range(theta, phi, C_dir=0.0, SINR_threshold=SINR_threshold, jammer_pos=jammer_pos) for theta in theta_range_iso]
    
    # Case 3: Directed without jammer (C_dir=1, no jammer) - LIMITED RANGE
    ranges_dir_no_jam = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, jammer_pos=None) for theta in theta_range_dir]
    
    # Case 4: Directed with jammer (C_dir=1, with jammer) - LIMITED RANGE
    ranges_dir_with_jam = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, jammer_pos=jammer_pos) for theta in theta_range_dir]
    
    # Convert theta to actual angles (relative to antenna orientation phi)
    actual_angles_iso = theta_range_iso + phi
    actual_angles_dir = theta_range_dir + phi
    
    # Convert to Cartesian coordinates for all cases
    range_x_iso_no_jam = p_tx[0] + np.array(ranges_iso_no_jam) * np.cos(actual_angles_iso)
    range_y_iso_no_jam = p_tx[1] + np.array(ranges_iso_no_jam) * np.sin(actual_angles_iso)
    
    range_x_iso_with_jam = p_tx[0] + np.array(ranges_iso_with_jam) * np.cos(actual_angles_iso)
    range_y_iso_with_jam = p_tx[1] + np.array(ranges_iso_with_jam) * np.sin(actual_angles_iso)
    
    range_x_dir_no_jam = p_tx[0] + np.array(ranges_dir_no_jam) * np.cos(actual_angles_dir)
    range_y_dir_no_jam = p_tx[1] + np.array(ranges_dir_no_jam) * np.sin(actual_angles_dir)
    
    range_x_dir_with_jam = p_tx[0] + np.array(ranges_dir_with_jam) * np.cos(actual_angles_dir)
    range_y_dir_with_jam = p_tx[1] + np.array(ranges_dir_with_jam) * np.sin(actual_angles_dir)
    
    # Create figure
    plt.figure(figsize=(27, 12))
    
    # Fix matplotlib deprecation warning - use new colormap access method
    colors = [plt.colormaps['viridis'](i) for i in [0.0, 0.33, 0.66, 1.0]]  # 4 evenly spaced colors
    
    # Plot all four communication range patterns with viridis colors
    fill_alpha = 0.3  # Increased alpha for better visibility with viridis
    inj_order = 1
    plt.fill(range_x_iso_no_jam, range_y_iso_no_jam, alpha=fill_alpha, color=colors[0], 
             label=fr'$C_{{\text{{dir}}}}=0,C_{{\text{{jam}}}}=0$', zorder=inj_order)
    plt.plot(range_x_iso_no_jam, range_y_iso_no_jam, color=colors[0], linewidth=linewidth, zorder=inj_order)
    
    dnj_order = 2
    plt.fill(range_x_dir_no_jam, range_y_dir_no_jam, alpha=fill_alpha, color=colors[1], 
             label=fr'$C_{{\text{{dir}}}}=1,C_{{\text{{jam}}}}=0$', zorder=dnj_order)
    plt.plot(range_x_dir_no_jam, range_y_dir_no_jam, color=colors[1], linewidth=linewidth, zorder=dnj_order)
    
    ij_order = 3
    plt.fill(range_x_iso_with_jam, range_y_iso_with_jam, alpha=fill_alpha, color=colors[2], 
             label=fr'$C_{{\text{{dir}}}}=0,C_{{\text{{jam}}}}=3$', zorder=ij_order)
    plt.plot(range_x_iso_with_jam, range_y_iso_with_jam, color=colors[2], linewidth=linewidth, zorder=ij_order)
    
    dj_order = 4
    plt.fill(range_x_dir_with_jam, range_y_dir_with_jam, alpha=fill_alpha, color=colors[3], 
             label=fr'$C_{{\text{{dir}}}}=1,C_{{\text{{jam}}}}=3$', zorder=dj_order)
    plt.plot(range_x_dir_with_jam, range_y_dir_with_jam, color=colors[3], linewidth=linewidth, zorder=dj_order)


    # Plot receiving agent - using custom marker size
    receiver_color = 'black'
    transmitter_color = 'black'
    plt.plot(p_rec[0], p_rec[1], 'o', color=receiver_color, markersize=marker_size, zorder=6)
    
    # Plot transmitter (changed to blue) - using custom marker size
    plt.plot(p_tx[0], p_tx[1], 'o', color=transmitter_color, markersize=marker_size, zorder=6)
    
    # Draw local x and y axes through transmitter
    axis_length = 1.1
    plt.plot([p_tx[0] - axis_length, p_tx[0] + axis_length], [p_tx[1], p_tx[1]], 'k-', linewidth=5, zorder=5)
    plt.text(axis_length + 0.1, 0, '', fontsize=12, ha='left')
    
    
    # Draw line from transmitter to receiver (changed to green) - using custom linewidth
    plt.plot([p_tx[0], p_rec[0]], [p_tx[1], p_rec[1]], '-', color=receiver_color, linewidth=linewidth, zorder=4)
    
    # Plot jammer - using custom marker size
    plt.plot(jammer_pos[0], jammer_pos[1], 'ro', markersize=marker_size, zorder=6)
    
    # Draw antenna orientation (phi) - using custom linewidth
    max_range_overall = max(max(ranges_dir_no_jam), max(ranges_dir_with_jam), 
                           max(ranges_iso_no_jam), max(ranges_iso_with_jam))
    antenna_length = 0.92*max_range_overall
    antenna_end = p_tx + antenna_length * np.array([np.cos(phi), np.sin(phi)])
    plt.arrow(p_tx[0], p_tx[1], antenna_end[0] - p_tx[0], antenna_end[1] - p_tx[1], 
              head_width=0.08, head_length=0.08, fc=transmitter_color, ec=transmitter_color, linewidth=linewidth, zorder=10)

              
    # Plot Rcom=1 circle around receiver - using custom linewidth
    circle_angles = np.linspace(0, 2*np.pi, 100)
    rcom_circle_x = p_rec[0] + 1.0 * np.cos(circle_angles)
    rcom_circle_y = p_rec[1] + 1.0 * np.sin(circle_angles)
    #plt.plot(rcom_circle_x, rcom_circle_y, 'g--', linewidth=linewidth, alpha=0.7, label=fr'Rcom=1 around $p_{{\text{{r}}}}$', zorder=3)

    
    # Create custom legend entry for boresight arrow - using custom linewidth
    #boresight_legend = Line2D([0], [0], color='blue', linewidth=linewidth, 
    #                     marker='>', markersize=8, markerfacecolor='blue', 
    #                     markeredgecolor='blue', label='Boresight')
    
    # Draw phi arc (changed to blue) - using custom linewidth
    phi_arc = np.linspace(0, phi, 50)
    arc_radius = 0.3
    phi_arc_x = p_tx[0] + arc_radius * np.cos(phi_arc)
    phi_arc_y = p_tx[1] + arc_radius * np.sin(phi_arc)
    plt.plot(phi_arc_x, phi_arc_y, transmitter_color, linewidth=linewidth-1, zorder=4)
    
    # Add arrow to phi arc (changed to blue)
    mid_phi = phi / 2  # Place arrow at midpoint of arc, not at the end
    arc_start = -8
    arrow_start = np.array([phi_arc_x[arc_start], phi_arc_y[arc_start]])
    arrow_dir = np.array([phi_arc_x[arc_start]-phi_arc_x[arc_start-1], phi_arc_y[arc_start]-phi_arc_y[arc_start-1]])
    plt.arrow(arrow_start[0], arrow_start[1], arrow_dir[0], arrow_dir[1], 
              head_width=0.02, head_length=0.02, fc=transmitter_color, ec=transmitter_color, zorder=4)
    
    # Phi text symbol - using custom symbol fontsize
    phi_text = arc_radius*np.array([np.cos(phi/2), np.sin(phi/2)]) + 0.03
    plt.text(phi_text[0], phi_text[1], rf'$\phi$', fontsize=symbol_fontsize, color=transmitter_color, zorder=10)
    
    # Draw theta_rec arc (changed to green)
    theta_rec = draw_theta_rec_arc(plt.gca(), p_tx, p_rec, phi, show_arrow=True)

    # Theta text symbol - using custom symbol fontsize
    theta_text = 0.6*np.array([np.cos(theta_rec/2+phi), np.sin(theta_rec/2+phi)]) + 0.03
    plt.text(theta_text[0]-0.01, theta_text[1]-0.04, rf'$\theta$', fontsize=symbol_fontsize, color=receiver_color, zorder=10)

    # Labels and annotations - using custom symbol fontsize
    text_label_gap = 0.08
    plt.text(p_rec[0] + 1.4*text_label_gap, p_rec[1] , rf'$p_{{\text{{r}}}}$', fontsize=symbol_fontsize, ha='left')
    plt.text(p_tx[0] - 2*text_label_gap, p_tx[1] - 2*text_label_gap, rf'$p_{{\text{{t}}}}$', fontsize=symbol_fontsize, ha='left')
    plt.text(jammer_pos[0] - 2*text_label_gap, jammer_pos[1] - 2*text_label_gap, rf'$p_{{\text{{j}}}}$', fontsize=symbol_fontsize, ha='left')

    
    # Conditional text boxes
    if include_text_boxes:
        # Phi text box (moved to right)
        plt.text(0.75, 0.80, rf'$\phi = {phi:.2f}$ rad', fontsize=12, color='blue', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # Theta text box (moved to right)
        plt.text(0.75, 0.70, rf'$\theta_{{rec}} = {theta_rec:.2f}$ rad', fontsize=12, color='green', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                transform=plt.gca().transAxes, verticalalignment='top')
        
        # Add formulas in text box (moved to right)
        formula_text = (r'$R(\theta) = \sqrt{\frac{|a(0)^H a(\theta)|}{SINR_0(1+C_{jam}\|p_{rec}-p_{jam}\|^{-2})}}$' + '\n' +
                       r'Isotropic: $\theta \in [-\pi, \pi]$, Directed: $\theta \in [-\pi/2, \pi/2]$')
        
        plt.text(0.75, 0.98, formula_text, transform=plt.gca().transAxes, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
                 verticalalignment='top')
        
        # Add comparison statistics and position info in text box (moved to right)
        max_ranges = {
            'Iso, No Jam': max(ranges_iso_no_jam),
            'Iso, With Jam': max(ranges_iso_with_jam),
            'Dir, No Jam': max(ranges_dir_no_jam),
            'Dir, With Jam': max(ranges_dir_with_jam)
        }
        
        # Position information
        receiver_distance = np.linalg.norm(p_rec - p_tx)
        jammer_distance = np.linalg.norm(jammer_pos - p_rec)
        
        stats_text = 'Position Info:'
        stats_text += f'\nReceiver: r={receiver_distance:.2f}, θ={receiver_angle:.2f} rad'
        stats_text += f'\nJammer: r={jammer_distance:.2f}, θ={jammer_angle_from_receiver:.2f} rad'
        stats_text += '\n\nMax Ranges:'
        for case, max_range in max_ranges.items():
            stats_text += f'\n{case}: {max_range:.2f}'
        
        plt.text(0.75, 0.60, stats_text, transform=plt.gca().transAxes, fontsize=9, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                 verticalalignment='top', horizontalalignment='left')
    
    #plt.grid(True, alpha=0.3)
    
    plt.axis('off') 
    # Set limits with controlled x-axis extension
    plt.xlim([-2.8, 1.6])  # Changed from [-3, 1.5] to [-2, 2] for more balanced view
    plt.ylim([-1.1, 1.6])

    # Set equal aspect ratio while preserving the limits
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    
    # Create legend with higher alpha for fill areas but keep everything else the same
    from matplotlib.patches import Patch
    
    # Get existing legend handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create custom legend patches with higher alpha for communication ranges only
    legend_alpha = 0.9  
    
    # Print debug info to see what labels we actually have
    print("Available labels:", labels)
    
    # Find and replace the communication range entries with higher alpha versions
    new_handles = []
    for handle, label in zip(handles, labels):
        # Match the exact labels from your plotting code
        if 'C_{\\text{dir}}=0,C_{\\text{jam}}=0' in label:
            new_handles.append(Patch(color=colors[0], alpha=legend_alpha, label=label))
            print(f"Replaced: {label}")
        elif 'C_{\\text{dir}}=1,C_{\\text{jam}}=0' in label:
            new_handles.append(Patch(color=colors[1], alpha=legend_alpha, label=label))
            print(f"Replaced: {label}")
        elif 'C_{\\text{dir}}=0,C_{\\text{jam}}=3' in label:
            new_handles.append(Patch(color=colors[2], alpha=legend_alpha, label=label))
            print(f"Replaced: {label}")
        elif 'C_{\\text{dir}}=1,C_{\\text{jam}}=3' in label:
            new_handles.append(Patch(color=colors[3], alpha=legend_alpha, label=label))
            print(f"Replaced: {label}")
        else:
            # Keep all other handles unchanged (transmitter, receiver, jammer, etc.)
            new_handles.append(handle)
            print(f"Kept original: {label}")

    # Create legend positioned in upper left with the new handles - using custom legend fontsize
    plt.legend(handles=new_handles, loc='center left', 
          frameon=False, fontsize=legend_fontsize)

    
    # Save plot if requested
    if save_plot:
        plot_dir = "Plots/communication"
        plot_name = "communication_range_comparison_all_cases"
        os.makedirs(plot_dir, exist_ok=True)

        # Save as high-quality PDF (vector format) for LaTeX
        pdf_path = os.path.join(plot_dir, f"{plot_name}.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        
        # Also save as high-resolution PNG as backup
        png_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
        
        print(f"Plot saved to: {pdf_path} and\n {png_path}")
    else:
        plt.show()
    
    # Print numerical comparison only if text boxes are included
    if include_text_boxes:
        # Calculate stats for printing
        max_ranges = {
            'Iso, No Jam': max(ranges_iso_no_jam),
            'Iso, With Jam': max(ranges_iso_with_jam),
            'Dir, No Jam': max(ranges_dir_no_jam),
            'Dir, With Jam': max(ranges_dir_with_jam)
        }
        receiver_distance = np.linalg.norm(p_rec - p_tx)
        jammer_distance = np.linalg.norm(jammer_pos - p_rec)
        
        print("\n" + "="*70)
        print("COMMUNICATION RANGE COMPARISON")
        print("="*70)
        print(f"SINR threshold: {SINR_threshold}")
        print(f"Antenna orientation (φ): {phi:.3f} rad = {np.degrees(phi):.1f}°")
        print(f"Receiver position: distance={receiver_distance:.3f}, angle={receiver_angle:.3f} rad = {np.degrees(receiver_angle):.1f}°")
        print(f"Jammer position (from receiver): distance={jammer_distance:.3f}, angle={jammer_angle_from_receiver:.3f} rad = {np.degrees(jammer_angle_from_receiver):.1f}°")
        print(f"Receiver angle (θ_rec): {theta_rec:.3f} rad = {np.degrees(theta_rec):.1f}°")
        print("-"*70)
        
        for case, max_range in max_ranges.items():
            print(f"{case:20s}: Max Range = {max_range:.3f}")
        
        print("-"*70)
        print("Improvement factors (relative to isotropic with jammer):")
        baseline = max_ranges['Iso, With Jam']
        for case, max_range in max_ranges.items():
            if case != 'Iso, With Jam':
                improvement = max_range / baseline
                print(f"{case:20s}: {improvement:.2f}x")
        print("="*70)

def plot_all_directed_lobes(save_plot=False, SINR_threshold=1.0, 
                           receiver_range=np.sqrt(2)/2, receiver_angle=np.pi/4,
                           marker_size=12, symbol_fontsize=16, 
                           linewidth=3, legend_fontsize=12):
    """
    Plot all possible directed lobe positions in one plot.
    Shows directed communication patterns for k*π/4, where k=0,1,2,...,7
    """
    
    # Setup
    p_tx = np.array([0, 0])  # Transmitter at origin
    p_rec = p_tx + receiver_range * np.array([np.cos(receiver_angle), np.sin(receiver_angle)])
    
    # Define all phi values: k*π/4 for k=0,1,2,...,7
    phi_values = [k * np.pi / 4 for k in range(8)]
    phi_labels = [f'φ={k}π/4' for k in range(8)]
    
    # Create figure
    plt.figure(figsize=(16, 16))
    
    # Get colormap for different lobes
    colors = [plt.colormaps['tab10'](i) for i in range(8)]
    
    # Plot all directed lobes (no jammer)
    fill_alpha = 0.15
    line_alpha = 0.8
    
    for i, (phi, label, color) in enumerate(zip(phi_values, phi_labels, colors)):
        # Create theta range for directed transmission
        theta_eps = 0.05
        theta_range_dir = np.linspace(-np.pi/2 + theta_eps, np.pi/2 - theta_eps, 500)
        
        # Calculate communication range for this phi (directed, no jammer)
        ranges_dir = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, 
                                        jammer_pos=None, p_tx=p_tx) for theta in theta_range_dir]
        
        # Convert theta to actual angles (relative to antenna orientation phi)
        actual_angles_dir = theta_range_dir + phi
        
        # Convert to Cartesian coordinates
        range_x_dir = p_tx[0] + np.array(ranges_dir) * np.cos(actual_angles_dir)
        range_y_dir = p_tx[1] + np.array(ranges_dir) * np.sin(actual_angles_dir)
        
        # Plot the communication range pattern
        plt.fill(range_x_dir, range_y_dir, alpha=fill_alpha, color=color, 
                label=label, zorder=i+1)
        plt.plot(range_x_dir, range_y_dir, color=color, linewidth=linewidth, 
                alpha=line_alpha, zorder=i+1)
        
        # Draw antenna boresight arrows for each direction
        max_range = max(ranges_dir) if ranges_dir else 1.0
        antenna_length = 0.8 * max_range
        antenna_end = p_tx + antenna_length * np.array([np.cos(phi), np.sin(phi)])
        
        plt.arrow(p_tx[0], p_tx[1], antenna_end[0] - p_tx[0], antenna_end[1] - p_tx[1], 
                  head_width=0.04, head_length=0.04, fc=color, ec=color, 
                  linewidth=linewidth-1, alpha=line_alpha, zorder=i+10)
    
    # Plot transmitter
    plt.plot(p_tx[0], p_tx[1], 'ko', markersize=marker_size+2, label='Transmitter', 
            zorder=20, markerfacecolor='white', markeredgewidth=2)
    
    # Plot receiver
    plt.plot(p_rec[0], p_rec[1], 'ks', markersize=marker_size, label='Receiver', 
            zorder=20, markerfacecolor='gray')
    
    # Draw line from transmitter to receiver
    plt.plot([p_tx[0], p_rec[0]], [p_tx[1], p_rec[1]], 'k--', 
            linewidth=linewidth-1, alpha=0.5, zorder=15)
    
    # Draw local coordinate axes
    axis_length = 1.2
    plt.arrow(p_tx[0], p_tx[1], axis_length, 0, head_width=0.03, head_length=0.03, 
              fc='black', ec='black', linewidth=1, zorder=15)
    plt.arrow(p_tx[0], p_tx[1], 0, axis_length, head_width=0.03, head_length=0.03, 
              fc='black', ec='black', linewidth=1, zorder=15)
    plt.text(axis_length + 0.05, 0, 'x', fontsize=symbol_fontsize-2, ha='left')
    plt.text(0, axis_length + 0.05, 'y', fontsize=symbol_fontsize-2, ha='center')
    
    # Add compass directions
    compass_radius = 1.4
    directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    for k, direction in enumerate(directions):
        angle = k * np.pi / 4
        x_pos = compass_radius * np.cos(angle)
        y_pos = compass_radius * np.sin(angle)
        plt.text(x_pos, y_pos, direction, fontsize=symbol_fontsize-4, 
                ha='center', va='center', 
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", alpha=0.8))
    
    # Labels and annotations
    text_label_gap = 0.05
    plt.text(p_rec[0] + text_label_gap, p_rec[1] + text_label_gap, 
            rf'$p_{{\text{{r}}}}$', fontsize=symbol_fontsize, ha='left', zorder=25)
    plt.text(p_tx[0] - 2*text_label_gap, p_tx[1] - 2*text_label_gap, 
            rf'$p_{{\text{{t}}}}$', fontsize=symbol_fontsize, ha='left', zorder=25)
    
    # Add formula in text box
    formula_text = (r'$R(\theta) = \sqrt{\frac{|a(0)^H a(\theta)|}{SINR_0}}$' + '\n' +
                   r'Directed: $C_{dir}=1$, $\theta \in [-\pi/2, \pi/2]$' + '\n' +
                   r'No jammer: $C_{jam}=0$')
    
    plt.text(0.02, 0.98, formula_text, transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
             verticalalignment='top')
    
    # Add statistics text box
    # Calculate max range for each direction
    max_ranges = []
    for phi in phi_values:
        theta_range_dir = np.linspace(-np.pi/2 + 0.05, np.pi/2 - 0.05, 100)
        ranges_dir = [communication_range(theta, phi, C_dir=1.0, SINR_threshold=SINR_threshold, 
                                        jammer_pos=None, p_tx=p_tx) for theta in theta_range_dir]
        max_ranges.append(max(ranges_dir) if ranges_dir else 0.0)
    
    stats_text = 'Max Ranges:'
    for i, (label, max_range) in enumerate(zip(phi_labels, max_ranges)):
        stats_text += f'\n{label}: {max_range:.2f}'
    
    overall_max = max(max_ranges)
    stats_text += f'\n\nOverall Max: {overall_max:.2f}'
    stats_text += f'\nSINR threshold: {SINR_threshold}'
    
    plt.text(0.02, 0.65, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
             verticalalignment='top', horizontalalignment='left')
    
    # Grid and formatting
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and limits
    max_extent = max(max_ranges) * 1.1
    plt.xlim(-max_extent, max_extent)
    plt.ylim(-max_extent, max_extent)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Title and labels
    plt.xlabel('x', fontsize=symbol_fontsize-2)
    plt.ylabel('y', fontsize=symbol_fontsize-2)
    plt.title(f'All Directed Communication Lobes (φ = k·π/4, k=0,1,...,7)\n' + 
              f'SINR threshold = {SINR_threshold}, No Jammer', fontsize=16)
    
    # Create legend with two columns to fit all 8 directions plus transmitter/receiver
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.35), 
              fontsize=legend_fontsize-2, ncol=2, frameon=True, fancybox=True, 
              shadow=True)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_dir = "Plots/communication"
        os.makedirs(plot_dir, exist_ok=True)
        plot_name = "all_directed_lobes"
        
        # Save as high-quality PDF
        pdf_path = os.path.join(plot_dir, f"{plot_name}.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', transparent=False)
        
        # Save as high-resolution PNG
        png_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
        
        print(f"Plot saved to: {pdf_path} and {png_path}")
    
    plt.show()
    
    # Print numerical summary
    print("\n" + "="*60)
    print("DIRECTED LOBE ANALYSIS")
    print("="*60)
    print(f"SINR threshold: {SINR_threshold}")
    print(f"Directivity: C_dir = 1.0 (directed)")
    print(f"Jammer: None (C_jam = 0)")
    print("-"*60)
    
    for i, (label, max_range) in enumerate(zip(phi_labels, max_ranges)):
        angle_deg = i * 45  # Convert to degrees
        print(f"{label:8s} ({angle_deg:3d}°): Max Range = {max_range:.3f}")
    
    print("-"*60)
    print(f"Overall Maximum Range: {max(max_ranges):.3f}")
    print(f"Overall Minimum Range: {min(max_ranges):.3f}")
    print(f"Range Variation: {max(max_ranges) - min(max_ranges):.3f}")
    print("="*60)



        

if __name__ == "__main__":
    # Set publication parameters
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'serif',
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'grid.linewidth': 0.5
    })
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Computer Modern']
    #plt.rcParams['text.usetex'] = True





    # Communication range comparison plot with custom sizes
    plot_communication_range_comparison(
        phi=np.pi/5,
        receiver_range=np.sqrt(2)/2,
        receiver_angle=np.pi/16,
        jammer_range_from_receiver=0.7,
        jammer_angle_from_receiver=-np.pi/3,
        include_text_boxes=False,
        save_plot=True,
        marker_size=30,        # Larger markers
        symbol_fontsize=45,    # Larger symbols (φ, θ, p_t, p_r)
        linewidth=6,           # Thicker lines
        legend_fontsize=40     
    )

    
    # Plot all directed lobe positions
    """
    plot_all_directed_lobes(
        save_plot=True,
        SINR_threshold=1.0,
        receiver_range=np.sqrt(2)/2,
        receiver_angle=np.pi/4,
        marker_size=12,
        symbol_fontsize=16,
        linewidth=3,
        legend_fontsize=12
    )
    """
    
    # Example with smaller elements for dense plots
    #plot_communication_range_comparison(
    #    phi=np.pi/3,
    #    receiver_range=np.sqrt(2)/2,
    #    receiver_angle=np.pi/4,
    #    jammer_range_from_receiver=1.0,
    #    jammer_angle_from_receiver=-np.pi/4,
    #    include_text_boxes=False,
    #    marker_size=8,         # Smaller markers
    #    symbol_fontsize=12,    # Smaller symbols
    #    linewidth=2,           # Thinner lines
    #    legend_fontsize=10     # Smaller legend
    #)
