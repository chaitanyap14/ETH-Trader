import sys
import numpy as np
import pandas as pd
from pathlib import Path
from agent.dqn_agent import DQNAgent, RuleBasedAgent
from environment.trading_env import TradingEnv

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

def preprocess_data(data):
    return data

def test_model():
    data = pd.read_pickle('./data/processed/test.pkl')
    data = preprocess_data(data)
    
    env = TradingEnv(data)
    agent = RuleBasedAgent(env.lookback_window, 3)
    agent.load('./agent/saved/final')
    
    state = env.reset()
    total_profit = 0
    
    for i in tqdm(range(len(data))):
        agent.epsilon = 0
        action = agent.act(state)
        next_state, reward, done, executed_state, executed_action = env.step(action)
        total_profit += reward
        state = next_state
        
        if done:
            print(f"Total Profit: {total_profit:.2f}")
            print(f"Account Balanec: {env.account_balance:.2f}")
            break

if __name__ == "__main__":
    test_model()

