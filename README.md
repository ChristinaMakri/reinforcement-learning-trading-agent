# Deep Reinforcement Learning Trading Agent with Synthetic Data

This project implements a Bitcoin trading agent using Deep Q-Learning (DQN) enhanced with synthetic data containing rare events such as crashes, spikes, and volatility segments.

---

## Contents

- `data.py`: Download and preprocess real and synthetic Bitcoin data.
- `environment.py`: Trading environment definition.
- `agent.py`: DQN agent implementation using PyTorch.
- `train.py`: Training the agent with and without synthetic data.
- `evaluate.py`: Agent evaluation.
- `visualize.py`: Visualization of training results.
- `main.py`: Main script running the complete training and evaluation pipeline.

---

## Usage Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the main script:
   ```bash
   python main.py
   ```

## Notes

-Uses the yfinance library to fetch historical Bitcoin data.  
-Synthetic data is generated simulating prolonged crashes, flash crashes, and volatile periods.  
-The model is trained with Deep Q-Learning via PyTorch.  
-Provides visualization of agent performance during training.  
-Suitable for experimentation with reinforcement learning on financial data.
