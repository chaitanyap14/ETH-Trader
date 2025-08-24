import sys
import numpy as np
import pandas as pd
from pathlib import Path
from agent.dqn_agent import DQNAgent
from agent.dqn_agent import DTAgent
from agent.dqn_agent import RuleBasedAgent
from environment.trading_env import TradingEnv
import tensorflow as tf

from multiprocessing import Pool, Manager
import multiprocessing as mp
from collections import deque
from tqdm import tqdm
import os
from functools import partial

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def preprocess_data(data):
    # Add your technical indicators here
    #data = data.tail(52704)
    return data

def main():
    # Load and prepare data
    data = pd.read_pickle('./data/processed/train.pkl')
    data = preprocess_data(data)
    print(data.columns)
    
    # Initialize environment and agent
    env = TradingEnv(data)
    agent = RuleBasedAgent(
        state_size=env.lookback_window,
        action_size=3  # 0: Long, 1: Short, 2: Do Nothing.
    )
    
    ## Training parameters
    #episodes = 1000
    #update_frequency = 500

    #
    ## Training loop
    #for episode in range(episodes):
    #    try:
    #        state = env.reset()
    #        total_reward = 0
 
    #        #248998
    #        for i in tqdm(range(52705), leave=False):
    #            action = agent.act(state)
    #            next_state, reward, done = env.step(action)
    #            agent.remember(state, action, reward, next_state, done)

    #            if i % update_frequency == 0:
    #                agent.replay()


    #            #if(reward != 0):
    #            #    print(reward)
    #                        
    #            total_reward += reward
    #            state = next_state

    #            if done:
    #                print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}, Balance: {env.account_balance:.2f}, epsilon: {agent.epsilon:.4f}")
    #                break

    episodes = 1000
    update_frequency = 250  # Update rules every N steps

    # Tracking best performance (EXISTING FUNCTIONALITY)
    best_balance = 0
    best_reward = -np.inf
    best_episode = 0

    for episode in range(episodes):
        try:
            state = env.reset()
            total_reward = 0
            done = False

            with tqdm(total=249900, desc=f"Episode {episode+1}", leave=False) as pbar:
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, executed_state, executed_action = env.step(action)
                    if executed_action is not None and executed_state is not None:
                        if action==2 or (action!=2 and reward!=0):
                            #print(executed_action, reward)
                            agent.remember(executed_state, executed_action, reward, next_state, done)
                    total_reward += reward
                    state = next_state
                    pbar.update(1)

                    # Periodic rule updates (NEW)
                    if episode <= 3:
                        agent.epsilon = 1

                    if env.current_step % update_frequency == 0:
                        #agent.replay()
                        agent.update_rules()

            # Update rules with episode return (NEW)
            agent.update_rules(episode_return=total_reward)

            current_balance = env.account_balance

            # Checkpoint saving (EXISTING FUNCTIONALITY)
            if current_balance > best_balance*1.01:
                best_balance = current_balance
                best_reward = total_reward
                best_episode = episode
                if agent.epsilon <= 0.2:
                    agent.save(f'./agent/saved/best_balance_{best_balance:.2f}')
                print(f"New best balance: {best_balance:.2f}")

            # Debug output (ENHANCED)
            print(f"\nEpisode {episode+1}: "
                  f"Reward={total_reward:.2f} (Best: {best_reward:.2f}), "
                  f"Balance={current_balance:.2f} (Best: {best_balance:.2f}), "
                  f"ε={agent.epsilon:.3f}, "
                  f"Rules={len(agent.rule_book)}") 

            # Print top rules every 10 episodes (NEW)
            if (episode+1) % 10 == 0:
                agent.print_top_rules(3)
                print(f"Best balance: {best_balance:.2f} (Episode {best_episode+1})")

            print(f"\nEpisode {episode+1} Metrics:")
            print(f"- Rules: {len(agent.rule_book)}")
            print(f"- Coverage: {100*len(agent.rule_book)/(2**18):.1f}% of possible states")  # 2^num_features
            print(f"- Avg Profit Factor: {np.mean([r[1] for r in agent.rule_book.values()]):.2f}")

        except KeyboardInterrupt:
            print("\nTraining stopped by user")
            break

    # Save final model (EXISTING)
    agent.save('./agent/saved/final')
    print(f"\nTraining completed. Best balance: {best_balance:.2f}")
    print("Top 5 rules:")
    agent.print_top_rules(5)

if __name__ == "__main__":
    main()
