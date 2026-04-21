from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import random

app = Flask(__name__)
CORS(app)

# Environment parameters
base_demand = 100
price_sensitivity = 2
noise_range = (-5, 5)

# Price settings
min_price = 10
max_price = 100
price_step = 5
price_levels = list(range(min_price, max_price + 1, price_step))

# Q-learning parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500
steps_per_episode = 50

actions = [-1, 0, 1]
num_states = len(price_levels) * 3
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))
is_trained = False


def get_demand_level(demand):
    if demand < 30:
        return 0
    if demand < 70:
        return 1
    return 2


def get_demand(price):
    noise = random.randint(noise_range[0], noise_range[1])
    return max(0, base_demand - price_sensitivity * price + noise)


def get_reward(price, demand):
    revenue = price * demand
    penalty = 50 if demand < 20 else 0
    return revenue - penalty


def get_state_index(price_idx, demand_level):
    return price_idx * 3 + demand_level


def train_q_model():
    global is_trained

    if is_trained:
        return

    for _ in range(episodes):
        price_idx = random.randint(0, len(price_levels) - 1)

        for _ in range(steps_per_episode):
            price = price_levels[price_idx]
            demand = get_demand(price)
            demand_level = get_demand_level(demand)

            state = get_state_index(price_idx, demand_level)

            if random.uniform(0, 1) < epsilon:
                action_idx = random.randint(0, num_actions - 1)
            else:
                action_idx = int(np.argmax(Q[state]))

            action = actions[action_idx]
            new_price_idx = max(0, min(len(price_levels) - 1, price_idx + action))

            new_price = price_levels[new_price_idx]
            new_demand = get_demand(new_price)
            new_demand_level = get_demand_level(new_demand)

            reward = get_reward(new_price, new_demand)
            new_state = get_state_index(new_price_idx, new_demand_level)

            Q[state, action_idx] = Q[state, action_idx] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action_idx]
            )

            price_idx = new_price_idx

    is_trained = True


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/simulate")
def simulate():
    train_q_model()

    payload = request.get_json(silent=True) or {}
    initial_price = int(payload.get("initial_price", 50))
    steps = int(payload.get("steps", 30))

    if (
        initial_price < min_price
        or initial_price > max_price
        or (initial_price - min_price) % price_step != 0
    ):
        return (
            jsonify(
                {
                    "error": "initial_price must be between 10 and 100 using increments of 5"
                }
            ),
            400,
        )

    if steps <= 0 or steps > 300:
        return jsonify({"error": "steps must be between 1 and 300"}), 400

    try:
        price_idx = price_levels.index(initial_price)
    except ValueError:
        return jsonify({"error": "invalid initial_price"}), 400

    rows = []

    for step in range(1, steps + 1):
        price = price_levels[price_idx]
        demand = get_demand(price)
        demand_level = get_demand_level(demand)

        state = get_state_index(price_idx, demand_level)
        action_idx = int(np.argmax(Q[state]))
        action = actions[action_idx]

        rows.append(
            {
                "step": step,
                "price": price,
                "demand": int(demand),
                "action": action,
            }
        )

        price_idx = max(0, min(len(price_levels) - 1, price_idx + action))

    return jsonify({"rows": rows})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
