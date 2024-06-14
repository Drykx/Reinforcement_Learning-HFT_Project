# RL in Limit Order Book for Cryptocurrency Trading

## Introduction

In the course Reinforcement Learning `EE-568` at EPFL, under the supervision of Yves Rechner and Philipp J. Schneider, we trained trading agents with **PPO** and **StableBaselines3** based on [mbt_gym](https://github.com/JJJerome/mbt_gym) module, a suite of gymnasium environment who aims to solve model-based HFT problems such as market-making and optimal execution.

### Abstract 

### Research Questions

## Organization

### Github Organisation

You can find our modification of the [mbt_gym](mbt_gym) module with the main input being the [vizualisation methods](mbt_gym/gym/helpers/helper2.py) and the new functions in certain files to implement the backtest; here is an example of one our [Jupyter Notebooks](CleanExample.ipynb) with this changes. 

The [Report](src/Project.pdf) and the [Poster](src/Poster.pdf) are in the source folder.

### Reproducibility 

The code only runs with a Python version prior to 3.11, i.e. 3.10.9. 

#### Setup Instructions

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Drykx/Reinforcement_Learning-HFT_Project.git
   cd Reinforcement_Learning-HFT_Project
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   .\venv\Scripts\activate   # On Windows
   pip install -r requirements.txt
   ```

## Improvement



