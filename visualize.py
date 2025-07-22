import matplotlib.pyplot as plt

def plot_rewards(rewards_real, rewards_synth):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_real, label='Χωρίς Synthetic Data')
    plt.plot(rewards_synth, label='Με Synthetic Data')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Comparison During Training')
    plt.legend()
    plt.grid(True)
    plt.show()
