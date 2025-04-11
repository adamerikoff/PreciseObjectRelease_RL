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
    # pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)

    env = environment.Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_size, device=device, update_every=UPDATE_EVERY_EPS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, lr=LR, tau=TAU)

    # Training tracking
    last_10_rewards = deque(maxlen=10)
    last_100_rewards = deque(maxlen=100)
    last_100_success = deque(maxlen=100)
    last_100_times = deque(maxlen=100)
    last_100_rtimes = deque(maxlen=100)
    last_100_steps = deque(maxlen=100)
    real_time_elapsed = 0.0
    total_training_time = 0.0
    epsilon = EPSILON_START

    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        'episode', 'reward', 'steps', 'avg10', 'avg100', 'avg100success',
        'epsilon', 'time_elapsed', 'real_time_elapsed', 'timestamp'
    ])

    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        state = env.reset(phi=PHI)
        episode_reward = 0
        episode_steps = 0
        done = False
        success = False
        start_time = time.perf_counter()

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
            episode_reward += reward
            episode_steps += 1
            agent.step(state, action, reward, next_state, done)
            state = next_state
            real_time_elapsed += PHYSICS_DT

            # env.render()

        success = env.success

        # Update tracking
        time_elapsed = time.perf_counter() - start_time
        total_training_time += time_elapsed
        last_100_rewards.append(episode_reward)
        last_10_rewards.append(episode_reward)
        last_100_success.append(1 if success else 0)
        last_100_times.append(time_elapsed)
        last_100_rtimes.append(real_time_elapsed)
        last_100_steps.append(episode_steps)

        # Calculate averages
        avg10 = np.mean(last_10_rewards)
        avg100 = np.mean(last_100_rewards) if episode >= 100 else np.nan
        avg100success = np.mean(last_100_success) * 100 if episode >= 100 else np.nan
        avgtime = np.mean(last_100_times) if episode >= 100 else np.nan
        avgrtime = np.mean(last_100_rtimes) if episode >= 100 else np.nan
        avgsteps = np.mean(last_100_steps) if episode >= 100 else np.nan

        # Print every episode (flushing)
        environment.print_episode(
            episode, episode_steps, episode_reward,
            epsilon, time_elapsed, real_time_elapsed
        )
        real_time_elapsed = 0.0

        # Every 100 episodes
        if episode % 100 == 0:
            # Detailed non-flushing print
            environment.print_100ep_summary(avgtime, avgrtime, avgsteps, avg100, total_training_time, epsilon, avg100success)

            # Add to DataFrame
            training_stats.loc[len(training_stats)] = {
                'episode': episode,
                'reward': episode_reward,
                'steps': avgsteps,
                'avg10': avg10,
                'avg100': avg100,
                'avg100success': avg100success,
                'epsilon': epsilon,
                'time_elapsed': time_elapsed,
                'real_time_elapsed': real_time_elapsed,
                'timestamp': datetime.now()
            }

        # Epsilon decay every episode (fixed placement)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # pr.close_window()

    # Final save and plot
    filename = 'training_stats_dqn_vanilla'
    training_stats.to_csv(f'{filename}.csv', index=False)
    
    training_stats = pd.read_csv(f"{filename}.csv")
    plot.plot_training_progress(training_stats, filename)

if __name__ == "__main__":
    main()