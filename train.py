from environment import TradingEnv
from agent import DQNAgent
from data import download_btc_data, add_synthetic_events_tagged, split_train_test

def train_agent(data, episodes=50, batch_size=32):
    env = TradingEnv(data)
    state_size = 3  # price, balance, position
    action_size = 3  # buy, sell, hold
    agent = DQNAgent(state_size, action_size)
    rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
        rewards.append(total_reward)
        print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    return agent, rewards


if __name__ == "__main__":
    # Download and prepare data
    btc_data = download_btc_data()
    train_data, test_data = split_train_test(btc_data)

    # Create synthetic training data
    synthetic_train_data = add_synthetic_events_tagged(train_data)

    print("Training without synthetic data...")
    agent_real, rewards_real = train_agent(train_data)

    print("\nTraining with synthetic data...")
    agent_synth, rewards_synth = train_agent(synthetic_train_data)

    # Εδώ μπορείς να σώσεις τα μοντέλα ή να κάνεις περαιτέρω αξιολόγηση
