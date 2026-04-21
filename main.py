import numpy as np
import random
import matplotlib.pyplot as plt

# Environment parameters
base_demand = 100
price_sensitivity = 2
noise_range = (-5, 5)

# Price settings
min_price = 10
max_price = 100
price_step = 5

price_levels = list(range(min_price, max_price + 1, price_step))

# Discretization function for demand
def get_demand_level(demand):
    if demand < 30:
        return 0  # Low
    elif demand < 70:
        return 1  # Medium
    else:
        return 2  # High

# Generate demand with noise
def get_demand(price):
    noise = random.randint(noise_range[0], noise_range[1])
    demand = max(0, base_demand - price_sensitivity * price + noise)
    return demand

# Reward function with penalty
def get_reward(price, demand):
    revenue = price * demand
    penalty = 0
    if demand < 20:
        penalty = 50  # discourage too high price
    return revenue - penalty

# Q-Learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

num_states = len(price_levels) * 3  # price levels * demand levels
num_actions = 3  # decrease, stay, increase

Q = np.zeros((num_states, num_actions))

# Map state to index
def get_state_index(price_idx, demand_level):
    return price_idx * 3 + demand_level

# Action mapping
actions = [-1, 0, 1]

# Training parameters
episodes = 500
steps_per_episode = 50

rewards_per_episode = []

for episode in range(episodes):
    price_idx = random.randint(0, len(price_levels) - 1)
    total_reward = 0

    for step in range(steps_per_episode):
        price = price_levels[price_idx]
        demand = get_demand(price)
        demand_level = get_demand_level(demand)

        state = get_state_index(price_idx, demand_level)

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, num_actions - 1)
        else:
            action_idx = np.argmax(Q[state])

        action = actions[action_idx]

        # Apply action
        new_price_idx = price_idx + action
        new_price_idx = max(0, min(len(price_levels) - 1, new_price_idx))

        new_price = price_levels[new_price_idx]
        new_demand = get_demand(new_price)
        new_demand_level = get_demand_level(new_demand)

        reward = get_reward(new_price, new_demand)
        total_reward += reward

        new_state = get_state_index(new_price_idx, new_demand_level)

        # Q-Learning update
        Q[state, action_idx] = Q[state, action_idx] + alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state, action_idx]
        )

        price_idx = new_price_idx

    rewards_per_episode.append(total_reward)

# Find optimal price
avg_rewards = []
for i, price in enumerate(price_levels):
    demand = get_demand(price)
    avg_rewards.append(get_reward(price, demand))

optimal_price = price_levels[np.argmax(avg_rewards)]

print("Optimal Price Learned:", optimal_price)

# Plot rewards
plt.plot(rewards_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()