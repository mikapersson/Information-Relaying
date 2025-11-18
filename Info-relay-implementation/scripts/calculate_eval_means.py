import pandas as pd

# Load CSV
df = pd.read_csv("/home/u099435/Info_relay_project/Information-Relaying/Info-relay-implementation/MAPPO_evaluation_results_K5_cpos0.5_cphi0.1_n10000_dir0_jam0.csv")

# Calculate averages
avg_value = df["value"].mean()
avg_agent_sum_distance = df["agent_sum_distance"].mean()
avg_air_distance = df["air_distance"].mean()
avg_delivery_time = df["delivery_time"].mean()

# Calculate success percentage
# assuming the column is boolean or strings like "True"/"False"
success_percentage = (df["sucess"] == True).mean() * 100

# Print results
print("=== Statistics ===")
print(f"Average value: {avg_value:.4f}")
print(f"Average agent_sum_distance: {avg_agent_sum_distance:.4f}")
print(f"Average air_distance: {avg_air_distance:.4f}")
print(f"Average delivery_time: {avg_delivery_time:.4f}")
print(f"Success percentage: {success_percentage:.2f}%")