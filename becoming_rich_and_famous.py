import numpy as np
import matplotlib.pyplot as plt

# Define the MDP
states = [0, 1, 2, 3]  # Four states  (PU, PF, RU, RF)
actions = [0, 1]  # Two actions per state
gamma = 0.9  # Discount factor

# Transition probabilities: P[s][a][s']
P = {
    0: {0: [1, 0, 0, 0], 1: [0.5, 0.5, 0, 0]},
    1: {0: [0.5, 0, 0.5, 0], 1: [0, 1, 0, 0]},
    2: {0: [0.5, 0, 0.5, 0], 1: [0.5, 0.5, 0, 0]},
    3: {0: [0, 0, 0.5, 0.5], 1: [0, 1, 0, 0]},
}

# Rewards: R[s][a]
R = {
    0: {0: 0, 1: -1},
    1: {0: 5, 1: -1},
    2: {0: 10, 1: -1},
    3: {0: 10, 1: -1},
}

# Value Iteration algorithm
def value_iteration(states, actions, P, R, gamma, nr_iterations=40, theta=1e-6):
    V = np.zeros(len(states))  # Initialize state values to zero
    policy = np.zeros(len(states), dtype=int)  # Initialize policy arbitrarily
    value_history = [V.copy()]  # To track value changes
    
    for i in range(nr_iterations):
        delta = 0
        for s in states:
            # Compute action-value function (for each state)
            action_values = []
            for a in actions:
                value = R[s][a] + gamma * sum(P[s][a][s_next] * V[s_next] for s_next in states)
                action_values.append(value)
            
            # Update the value of state s
            max_value = max(action_values)
            delta = max(delta, abs(max_value - V[s]))
            V[s] = max_value
            
            # Update the policy for state s
            policy[s] = np.argmax(action_values)
            
        # Track value function after each iteration
        value_history.append(V.copy())
        
        # Check for convergence
        if delta < theta:
            break
    
    return policy, V, value_history

# Solve the MDP
nr_iterations = 40
optimal_policy, optimal_values, value_history  = value_iteration(states, actions, P, R, gamma, nr_iterations)

""" # Plot the value function convergence
value_history = np.array(value_history)
iterations = range(len(value_history))

plt.figure(figsize=(8, 6))
for s in states:
    plt.plot(iterations, value_history[:, s], label=f"State {state_names[s]}")
plt.xlabel("Iteration")
plt.ylabel("Value Function")
plt.title("Value Function Convergence Over Iterations")
plt.legend()
plt.grid()
plt.show() """

# Sort states by the maximum value they achieve
value_history = np.array(value_history)
max_values = value_history.max(axis=0)
sorted_states = np.flip(np.argsort(max_values))

# Plot the value function convergence with sorted labels
plt.figure(figsize=(8, 6))
state_names = ["PU", "PF", "RU", "RF"]
for s in sorted_states:
    plt.scatter(range(len(value_history)), value_history[:, s], label=f"State {state_names[s]}")
plt.xlabel("Iteration")
plt.ylabel("Value Function")
plt.title("Value Function Convergence Over Iterations")
plt.legend()
plt.grid()
plt.show()

# Display the results
print("Optimal Policy:", optimal_policy)
print("Optimal Values:", optimal_values)



