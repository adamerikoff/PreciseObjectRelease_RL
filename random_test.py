import time
import pyray as pr
import environment
import numpy as np
from collections import deque
import pandas as pd
from datetime import datetime
from config import *

def main():
    # Initialize window and environment
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)
    env = environment.Environment((500, 400, 500))
    
    # Evaluation parameters
    num_episodes = 20
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
            # Choose random action (0 to 4 for the 5 possible actions)
            action = np.random.randint(0, 5)
            next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
            episode_reward += reward
            episode_steps_count += 1
            state = next_state
            
            env.render(top_view=True)
            time.sleep(0.01)
        
        success = env.success
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_steps_count)
        successes.append(success)
        
        print(f"Episode {episode}: Reward={episode_reward}, Steps={episode_steps_count}, Success={success}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    success_percentage = (np.sum(successes) / num_episodes * 100)
    
    print("\nRandom Action Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Percentage: {success_percentage:.2f}%")
    
    pr.close_window()

if __name__ == "__main__":
    main()