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

# Discretize demand
def get_demand_level(demand):
    if demand < 30:
        return 0
    elif demand < 70:
        return 1
    else:
        return 2

# Demand function with randomness
def get_demand(price):
    noise = random.randint(noise_range[0], noise_range[1])
    demand = max(0, base_demand - price_sensitivity * price + noise)
    return demand

# Reward function
def get_reward(price, demand):
    revenue = price * demand
    penalty = 0
    if demand < 20:
        penalty = 50
    return revenue - penalty

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

num_states = len(price_levels) * 3
num_actions = 3

Q = np.zeros((num_states, num_actions))

# State mapping
def get_state_index(price_idx, demand_level):
    return price_idx * 3 + demand_level

actions = [-1, 0, 1]

# Training
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

        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, num_actions - 1)
        else:
            action_idx = np.argmax(Q[state])

        action = actions[action_idx]

        new_price_idx = price_idx + action
        new_price_idx = max(0, min(len(price_levels) - 1, new_price_idx))

        new_price = price_levels[new_price_idx]
        new_demand = get_demand(new_price)
        new_demand_level = get_demand_level(new_demand)

        reward = get_reward(new_price, new_demand)
        total_reward += reward

        new_state = get_state_index(new_price_idx, new_demand_level)

        Q[state, action_idx] = Q[state, action_idx] + alpha * (
            reward + gamma * np.max(Q[new_state]) - Q[state, action_idx]
        )

        price_idx = new_price_idx

    rewards_per_episode.append(total_reward)

print("\nTraining Completed\n")


# Dynamic Pricing Simulation (SYSTEM)
print("Dynamic Pricing Simulation:\n")

price_idx = len(price_levels) // 2  # start from mid price

price_history = []
demand_history = []

for step in range(30):
    price = price_levels[price_idx]
    demand = get_demand(price)
    demand_level = get_demand_level(demand)

    state = get_state_index(price_idx, demand_level)

    # Use learned policy (no exploration)
    action_idx = np.argmax(Q[state])
    action = actions[action_idx]

    new_price_idx = price_idx + action
    new_price_idx = max(0, min(len(price_levels) - 1, new_price_idx))

    print(f"Step {step+1}: Price = {price}, Demand = {demand}, Action = {action}")

    price_history.append(price)
    demand_history.append(demand)

    price_idx = new_price_idx


# Plot Learning Curve (Smoothed)
window = 20
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(smoothed_rewards)
plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")
plt.title("Learning Curve (Smoothed)")
plt.show()


# Plot Price Trend (SYSTEM BEHAVIOR)
plt.figure()
plt.plot(price_history)
plt.xlabel("Steps")
plt.ylabel("Price")
plt.title("Dynamic Price Adjustment Over Time")
plt.show()