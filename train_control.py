import time
import pyray as pr
import environment
from dqn import DQNAgent
import numpy as np
import torch
from collections import deque

import pandas as pd
from datetime import datetime

def print_episode(episode, steps, reward, avg10, eps, time_elapsed, real_time_elapsed):
    """Flushing print for every episode"""
    print(
        f"Ep {episode:>6d} | "
        f"S:{steps:>4d} | "
        f"R:{reward:>7.2f} | "
        f"Avg10:{avg10:>7.2f} | "
        f"ε:{eps:>.3f} | "
        f"T:{time_elapsed:>5.2f}s | " 
        f"RT:{real_time_elapsed:>5.2f}s", 
        end='\r',  # Use carriage return for in-place update
        flush=True
    )

def print_100ep_summary(avgtime, avgrtime, avgsteps, avg100, total_time, eps):
    """Non-flushing detailed print for every 100 episodes"""
    print("\n" + "="*80)
    print(f"- Last 100 Avg Time: {avgtime:.2f}s per episode")
    print(f"- Last 100 Avg Real Time: {avgrtime:.2f}s per episode")
    print(f"- Last 100 Avg Steps: {avgsteps:.2f}")
    print(f"- Last 100 Avg Reward: {avg100:.2f}")
    print(f"- Total Training Time: {total_time:.2f}s")
    print(f"- Current Epsilon: {eps:.4f}")
    print("="*80 + "\n")
    
def main():
    # Constants
    SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
    WINDOW_TITLE = "Drone Grenade Environment"
    PHYSICS_DT = 0.1
    N_EPISODES = 30_000
    
    # Training parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9999
    UPDATE_EVERY_EPS = 4
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 128
    # Initialize window and environment
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)

    env = environment.Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_size, device=device, update_every=UPDATE_EVERY_EPS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    
    # Training tracking
    last_100_rewards = deque(maxlen=100)
    last_10_rewards = deque(maxlen=10)
    last_100_times = deque(maxlen=100)
    last_100_rtimes = deque(maxlen=100)
    last_100_steps = deque(maxlen=100)
    real_time_elapsed = 0.0
    total_training_time = 0.0
    epsilon = EPSILON_START
    
    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        'episode', 'reward', 'steps', 'avg10', 'avg100', 
        'epsilon', 'time_elapsed', 'real_time_elapsed', 'timestamp'
    ])

    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        start_time = time.perf_counter()
        
        while not done and not pr.window_should_close():
            action = agent.act(state, epsilon)
            if action == 4:
                next_state, reward, done, steps = env.simulate_free_fall(PHYSICS_DT)
                episode_reward += reward
                episode_steps += steps
                agent.step(state, action, reward, next_state, done)
                state = next_state
                real_time_elapsed += PHYSICS_DT
            else:
                next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
                episode_reward += reward
                episode_steps += 1
                agent.step(state, action, reward, next_state, done)
                state = next_state
                real_time_elapsed += PHYSICS_DT

        # env.render()

        # Update tracking
        time_elapsed = time.perf_counter() - start_time
        total_training_time += time_elapsed
        last_100_rewards.append(episode_reward)
        last_10_rewards.append(episode_reward)
        last_100_times.append(time_elapsed)
        last_100_rtimes.append(real_time_elapsed)
        last_100_steps.append(episode_steps)
        
        # Calculate averages
        avg10 = np.mean(last_10_rewards)
        avg100 = np.mean(last_100_rewards) if episode >= 100 else np.nan
        avgtime = np.mean(last_100_times) if episode >= 100 else np.nan
        avgrtime = np.mean(last_100_rtimes) if episode >= 100 else np.nan
        avgsteps = np.mean(last_100_steps) if episode >= 100 else np.nan
        
        # Print every episode (flushing)
        print_episode(
            episode, episode_steps, episode_reward, 
            avg10, epsilon, time_elapsed, real_time_elapsed
        )
        real_time_elapsed = 0.0
        
        # Every 100 episodes
        if episode % 100 == 0:
            # Detailed non-flushing print
            print_100ep_summary(avgtime, avgrtime, avgsteps, avg100, total_training_time, epsilon)
            
            # Add to DataFrame
            training_stats.loc[len(training_stats)] = {
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'avg10': avg10,
                'avg100': avg100,
                'epsilon': epsilon,
                'time_elapsed': time_elapsed,
                'real_time_elapsed': real_time_elapsed,
                'timestamp': datetime.now()
            }

        # Epsilon decay every episode (fixed placement)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    pr.close_window()
    
    # Final save and plot
    training_stats.to_csv(f'training_stats.csv', index=False)
    
if __name__ == "__main__":
    main()