from data import download_btc_data, add_synthetic_events_tagged, split_train_test
from train import train_agent
from evaluate import evaluate_agent
from visualize import plot_rewards

def main():

    btc_data = download_btc_data()

    train_data, test_data = split_train_test(btc_data)
    synthetic_train_data = add_synthetic_events_tagged(train_data)


    agent_real, rewards_real = train_agent(train_data)


    agent_synth, rewards_synth = train_agent(synthetic_train_data)

 
    profit_real = evaluate_agent(agent_real, test_data)


    profit_synth = evaluate_agent(agent_synth, test_data)


    print(f"➤ Profit without Synthetic Data: {profit_real:.2f}")
    print(f"➤ Profit with Synthetic Data: {profit_synth:.2f}")

    plot_rewards(rewards_real, rewards_synth)

if __name__ == "__main__":
    main()
