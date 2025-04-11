import time
import pyray as pr
import environment
from dqn import DQNAgent
import numpy as np
import torch
from collections import deque
import plot
import pandas as pd
from datetime import datetime
from config import *

def main():
    # Initialize window and environment
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)
    env = environment.Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent
    agent = DQNAgent(env.state_size, env.action_size, device=device, 
                    update_every=UPDATE_EVERY_EPS, buffer_size=BUFFER_SIZE, 
                    batch_size=BATCH_SIZE, lr=LR, tau=TAU)
    
    # Load the trained model
    model_filename = 'dqn_jump_model.pth'
    try:
        agent.load(model_filename)
        print(f"Successfully loaded model from {model_filename}")
    except FileNotFoundError:
        print(f"Model file {model_filename} not found!")
        pr.close_window()
        return
    
    # Evaluate the model (set epsilon low for evaluation)
    num_episodes=20
    epsilon=0.05
    episode_rewards = []
    episode_steps = []
    successes = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset(phi=PHI)
        episode_reward = 0
        episode_steps_count = 0
        done = False
        success = False
        
        while not done:
            action = agent.act(state, epsilon)  # Use small epsilon for evaluation
            next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
            episode_reward += reward
            episode_steps_count += 1
            state = next_state
            
            env.render()
            time.sleep(0.01)
        
        success = env.success
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_steps_count)
        successes.append(success)
        
        print(f"Evaluation Episode {episode}: Reward={episode_reward}, Steps={episode_steps_count}, Success={success}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    success_percentage = (np.sum(successes) / num_episodes * 100)
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Percentage: {success_percentage:.2f}%")
    
    
    pr.close_window()
