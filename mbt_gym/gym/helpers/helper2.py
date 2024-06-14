import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment 
from mbt_gym.gym.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory




def plot_agent(observations,actions,rewards, bool):
    " Will generate the plot for the agent at a random trajectory if bool is True. If its false it will plot the worst one"

    if bool:
        j = np.random.randint(0, observations.shape[0])
    else:
        j = np.argmin(rewards[:, -1])  

    # Set Seaborn style to dark
    sns.set(style="darkgrid")

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))


    # Plot 3: Cash and inventory for a single trajectory
    axs[0, 0].plot(observations[j, 1, :], label='Inventory')
    axs[0, 0].set_title(f'Inventory process for trajectory {j}')
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].set_ylabel("Quantity assets held")

    # Plot 2: Cash and inventory for a single trajectory
    axs[0, 1].plot(observations[j, 0, :], label='Cash')
    axs[0, 1].set_title(f'Cash process for trajectory {j}')
    axs[0, 1].set_xlabel("Step")
    axs[0, 1].set_ylabel("Money")


    # Plot 3: Cumulative reward at each step
    axs[1, 0].plot(np.cumsum(rewards, axis=-1)[j])
    axs[1, 0].set_title(f"Cumulative reward of the agent for trajectory {j}")
    axs[1, 0].set_xlabel("Step")
    axs[1, 0].set_ylabel("Cumulative reward")


    # Plot 1: Actions for a single trajectory
    axs[1, 1].plot(observations[j, 3, :], label='$s_t$')
    axs[1, 1].plot(-actions[j,1,:] + observations[j, 3, :], alpha = 0.6, label="$p_{t}^{bid}$")
    axs[1, 1].plot(actions[j,0,:] + observations[j, 3, :] , alpha = 0.6, label="$p_{t}^{ask}$")
    axs[1, 1].set_title(f"Potential actions of the agent for trajectory {j}")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Price")
    axs[1, 1].legend()


    axs[0, 2].plot(observations[j, 4, :], label='Arrival Model State', color='b')
    axs[0, 2].set_ylabel("Number of times the midprice changes")
    axs[0, 2].set_xlabel("Step")
    axs[0, 2].set_title('Arrival stochastic process state')

    axs[1, 2].plot(observations[j, 5, :])
    axs[1, 2].set_title('Price impact state')
    axs[1,2].set_xlabel("Step")

    # Save the plot to a file
    plt.savefig('agent_plot.png', dpi=300, bbox_inches='tight', format='pdf')  # Save as PNG file with 300 dpi resolution

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_meanAgent(observations, actions, rewards):

    sns.set(style="darkgrid")

    # Calculate cumulative rewards
    cum_rewards = np.cumsum(rewards, axis=-1)
    mean_Crem = cum_rewards.mean(axis=0)
    sdev_Crem = cum_rewards.std(axis=0)

    # Calculate end money
    endmoney = observations[:, 1, :] * observations[:, 3, :] + observations[:, 0, :]
    mean_wealth = endmoney.mean(axis=0)
    sdev_PNL = endmoney.std(axis=0)

    # Create a 1x3 subplot
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    # Plot 1: Mean Cumulative Rewards with Standard Deviation
    axs[0].plot(mean_Crem, label="Mean Cumulative Rewards", color="b")
    axs[0].fill_between(np.arange(sdev_Crem.shape[0]), mean_Crem - sdev_Crem, mean_Crem + sdev_Crem, color='b', alpha=0.3, label="Standard Deviation")
    axs[0].set_title("Mean Cumulative Rewards with Standard Deviation")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Rewards")
    axs[0].legend()

    # Plot 2: Histogram of End Cumulative Rewards and PnL
    axs[1].hist(cum_rewards[:, -1], bins=80, density=True, alpha=0.6, color='b', label=f"Cumulative Reward, mean = {mean_Crem[-1].round(2)}, std = {sdev_Crem[-1].round(2)}")
    axs[1].hist(endmoney[:, -1], bins=80, density=True, alpha=0.3, color='r', label=f"End PnL, mean = {mean_wealth[-1].round(2)}, std = {sdev_PNL[-1].round(2)}")
    axs[1].set_title("Histogram of End Cumulative Reward and PnL")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Density")
    axs[1].set_xlim([-100, np.max(cum_rewards[:, -1]) + 10])
    axs[1].legend()

    # Plot 3: Mean PnL with Standard Deviation
    axs[2].plot(mean_wealth, label="Mean PnL", color="r")
    axs[2].fill_between(np.arange(sdev_PNL.shape[0]), mean_wealth - sdev_PNL, mean_wealth + sdev_PNL, color='r', alpha=0.3, label="Standard Deviation")
    axs[2].set_title("Mean PnL with Standard Deviation")
    axs[2].set_xlabel("Steps")
    axs[2].set_ylabel("Dollars")
    axs[2].legend()

    # Adjust layout
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig('MeanAgent_plot.png', dpi=300, bbox_inches='tight', format='pdf')  # Save as PNG file with 300 dpi resolution

    # Show the plot
    plt.show()


def plot_agent_BT(observations, actions, rewards, bool, sig = 100):
    "Will generate the plot for the agent at a random trajectory if bool is True. If it's false it will plot the worst one"

    # Instance
    endmoney = observations[:, 1, :] * observations[:, 3, :] + observations[:, 0, :]
    mean_wealth = endmoney.mean(axis=0)

    if bool:
        j = np.random.randint(0, observations.shape[0])
    else:
        j = np.argmin(rewards[:, -1])  

    # Set Seaborn style to dark
    sns.set(style="darkgrid")

    # Create a 2x2 grid layout
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 4, height_ratios=[1, 1])

    # Plot 1: Inventory process for a single trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(observations[j, 1, :], label='Inventory')
    ax1.set_title(f'Inventory process')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Quantity assets held")
    ax1.legend()

    # Plot 2: Cash process for a single trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(observations[j, 0, :], label='Cash')
    ax2.set_title(f'Cash process')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Money")
    ax2.legend()

    # Plot 3: Cumulative reward at each step
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.cumsum(rewards, axis=-1)[j])
    ax3.set_title(f"Cumulative reward")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Cumulative reward")

    # Plot 4: PnL at each step
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(mean_wealth, label="Mean PnL", color="r")
    ax4.set_title(f"PnL")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Dollars")
    ax4.legend()

    # Plot 5: Another plot (for example, the reward process)
    ax6 = fig.add_subplot(gs[1, 0:2])
    ax6.plot(gaussian_filter1d(actions[j, 1, :], sigma = sig) , alpha=0.9,color="green")
    ax6.plot(actions[j, 1, :]  , alpha=0.2, label="$\delta_{t}^{ask}$",color="green")
    ax6.plot(gaussian_filter1d(actions[j, 0, :], sigma= sig) , alpha=0.9,color="orange")
    ax6.plot(actions[j, 0, :]  , alpha=0.2, label="$\delta_{t}^{bid}$",color="orange")
    ax6.set_title(f'Actions Smoothed with $\sigma$ = {sig}')
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Dollars")
    ax6.legend(loc= "lower right")

    # Plot 6: Potential actions of the agent for trajectory
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.plot(observations[j, 3, :], label='$s_t$', alpha=1, color='b')
    ax5.plot(-actions[j, 1, :] + observations[j, 3, :], alpha=0.6, label="$p_{t}^{ask}$",color="green")
    ax5.plot(actions[j, 0, :] + observations[j, 3, :], alpha=0.8, label="$p_{t}^{bid}$",color="orange")
    ax5.set_title(f"Stock price")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Price")
    ax5.legend()

    # Save the plot to a file
    plt.savefig('agent_plot_characteristic.png', dpi=300, bbox_inches='tight', format='pdf')  # Save as PDF file with 300 dpi resolution

    # Adjust layout
    plt.tight_layout()
    plt.show()