import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar

# ----------------------------
# True optimal point on circle
# ----------------------------
def optimal_circle_boundary_point(p_k, p_r, p_c, R):
    """
    Compute the point on the circle boundary that minimizes
    ||p_k - p*|| + ||p* - p_r|| using 1D optimization over the circle angle.
    """

    if np.linalg.norm(p_k - p_c) < R:
        return p_k, 0.0
    else:
        def total_dist(theta):
            p_star = p_c + R * np.array([np.cos(theta), np.sin(theta)])
            return np.linalg.norm(p_k - p_star) + np.linalg.norm(p_r - p_star)
        
        res = minimize_scalar(total_dist, bounds=(0, 2*np.pi), method='bounded')
        theta_opt = res.x
        p_star = p_c + R * np.array([np.cos(theta_opt), np.sin(theta_opt)])
        return p_star, res.fun


if __name__ == "__main__":
    # ----------------------------
    # Setup
    # ----------------------------
    p_c = np.array([0.0, 0.0])   # Circle center
    R = 5.0                      # Circle radius
    p_r = np.array([6.0, -3.0])  # Fixed target point

    # Path for moving point p_k
    angles = np.linspace(0, 2*np.pi, 200)
    radius_k = 7.0
    p_k_path = np.stack([radius_k * np.cos(angles), radius_k * np.sin(angles)], axis=1)

    # Precompute total distances for all frames
    total_distances = []
    for i in range(len(p_k_path)):
        _, dist = optimal_circle_boundary_point(p_k_path[i], p_r, p_c, R)
        total_distances.append(dist)
    total_distances = np.array(total_distances)

    # Circle points for plotting
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = p_c[0] + R * np.cos(theta)
    circle_y = p_c[1] + R * np.sin(theta)

    # ----------------------------
    # Plot setup with two subplots
    # ----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: trajectory and geometry
    ax1.plot(circle_x, circle_y, 'b-', label='Circle')
    line1, = ax1.plot([], [], 'r--', label='p_k → p*')
    line2, = ax1.plot([], [], 'g--', label='p* → p_r')
    point_k, = ax1.plot([], [], 'ro', label='p_k')
    point_r, = ax1.plot(p_r[0], p_r[1], 'go', label='p_r')
    point_star, = ax1.plot([], [], 'mo', label='p* (optimal)')
    ax1.scatter(p_c[0], p_c[1], c='k', marker='x', label='Center')

    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_title('Optimal boundary point')

    # Right subplot: total distance over time
    ax2.plot(range(len(total_distances)), total_distances, 'b-', linewidth=2, label='Total distance')
    point_distance, = ax2.plot([], [], 'ro', markersize=8, label='Current frame')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Total Distance')
    ax2.set_title('||p_k - p*|| + ||p* - p_r||')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(0, len(total_distances)-1)
    ax2.set_ylim(min(total_distances)-0.5, max(total_distances)+0.5)

    # ----------------------------
    # Animation update function
    # ----------------------------
    def update(frame):
        p_k = p_k_path[frame, :]
        p_star, total_dist = optimal_circle_boundary_point(p_k, p_r, p_c, R)
        
        # Update left subplot
        line1.set_data([p_k[0], p_star[0]], [p_k[1], p_star[1]])
        line2.set_data([p_star[0], p_r[0]], [p_star[1], p_r[1]])
        point_k.set_data([p_k[0]], [p_k[1]])
        point_star.set_data([p_star[0]], [p_star[1]])
        
        # Update right subplot
        point_distance.set_data([frame], [total_dist])
        
        return line1, line2, point_k, point_star, point_distance

    # ----------------------------
    # Create animation
    # ----------------------------
    anim = FuncAnimation(fig, update, frames=len(p_k_path), interval=50, blit=True)

    # Save as GIF
    output_path = 'reflection_animation.gif'
    #anim.save(output_path, writer='pillow', fps=20, dpi=100)
    print(f"Animation saved to {output_path}")


    plt.tight_layout()
    plt.show()