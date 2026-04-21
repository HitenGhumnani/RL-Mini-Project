# Dynamic Pricing System using Reinforcement Learning

## Overview
This project implements a Dynamic Pricing System using Reinforcement Learning (Q-Learning). The system learns optimal pricing strategies by interacting with a simulated environment where demand varies with price. It dynamically adjusts prices over time to maximize revenue based on demand feedback.


## How It Works
- The system acts as a seller (agent) that decides product prices.
- Demand changes based on price (higher price → lower demand).
- The agent learns using Q-Learning:
  - Tries different prices  
  - Observes demand  
  - Receives reward (revenue)  
- Over time, it learns the best pricing strategy.


## RL Components
- **Agent:** Pricing system  
- **State:** Combination of price level and demand level  
- **Action:** Increase price, decrease price, or keep it same  
- **Reward:** Revenue (price × demand) with penalty for very low demand  


## Project Structure
- `api.py` → Backend (Flask API with RL model)  
- `index.html` → Frontend UI  
- `app.js` → Handles simulation and API calls  
- `styles.css` → Basic styling  


## How to Run

### 1. Install dependencies
```bash
pip install flask flask-cors numpy
```

### 2. Run backend
```bash
python api.py
```

### 3. Run frontend
- Open `index.html` in your browser
- Enter an initial price (between 10 and 100, step of 5)
- Click Start Simulation

## Output
- Step-by-step table showing:
  - Price
  - Demand
  - Action taken
- Graph showing price changes over time
