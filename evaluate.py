from environment import TradingEnv

def evaluate_agent(agent, data):
    env = TradingEnv(data)
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

    print(f"Evaluation total profit: {env.total_profit:.2f}")
    return env.total_profit
