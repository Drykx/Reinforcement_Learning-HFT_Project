# RL in Limit Order Book for Cryptocurrency Trading

## Introduction

In the course Reinforcement Learning `EE-568` at EPFL, under the supervision of Yves Rechner and Philipp J. Schneider, we trained trading agents with **PPO** and **StableBaselines3** based on [mbt_gym](https://github.com/JJJerome/mbt_gym) module, a suite of gymnasium environment who aims to solve model-based HFT problems such as market-making and optimal execution.

### Abstract 

Hawkes processes have been used to model the arrivals of orders in financial markets since the early 2000s. This model has gain popularity from its ability to capture the self-exciting nature of order flows—where an order can increase the likelihood of subsequent orders, creating clusters of activity that are characteristic of high-frequency trading environments.

We aimed to understand the dynamics of highly volatile cryptocurrency markets. Initially, we approached the problem from the perspective of a market maker. However, we discovered that the cluster of arrivals, modeled by Hawkes processes, presents a constant arbitrage opportunity. 
Specifically, our backtesting results showed that the agent would never hold or sell more than one asset. When the agent's order is executed within a cluster of filled orders, they would exit each order cluster with a positive spread bonus. 
We also demonstrated the agent's limited performance in excessively volatile environments, and as these clusters can only occur in moderately stochastic environments the agents performance will be poor in non volatile environment. 

### Research Questions

Our research questions were the following,

-   How is volatility of the mid-price influencing the performance of the model based agent (on backtest data)? 
-   Under which settings can we have the most adversial robust agent?
-   Can the performance bound if the last question be exceeded?

## Organization

### Github Organisation

You can find our modification of the [mbt_gym](mbt_gym) module with the main input being the [vizualisation methods](mbt_gym/gym/helpers/helper2.py) and the new functions in certain files to implement the backtest; here is an example of one our [Jupyter Notebooks](CleanExample.ipynb) with these changes. 

The [Report](src/Project.pdf) and the [Poster](src/Poster.pdf) are in the source folder.

### Reproducibility 

The code only runs with a Python version prior to 3.11, i.e. 3.10.9. 

#### Setup Instructions

   ```sh
   git clone https://github.com/Drykx/Reinforcement_Learning-HFT_Project.git
   cd Reinforcement_Learning-HFT_Project
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   .\venv\Scripts\activate   # On Windows
   pip install -r requirements.txt
   ```

## Future Work

It will be interesting to understand how time delay impacts the performance of the agent and which tunning has to be implemented to enhance the agent's performance. 

