from collections import deque
import time
from typing import Type


import torch
import torch.nn as nn

from config.config import BUFFER_SIZE, BATCH_SIZE, LR, TAU, EPSILON_START, N_EPISODES, PHYSICS_DT, EPSILON_END, EPSILON_DECAY, SCREEN_HEIGHT, SCREEN_WIDTH
from src.environment import Environment
from dqn.dqn_base import DQN
from dqn.buffer import Buffer
from src.renderer import Renderer

import pandas as pd 
import numpy as np

def train(
        title: str,
        q_network_class: Type[nn.Module],  # Your QNetwork class (Shallow/Medium/Deep)
        buffer_class: Buffer,  # Your ReplayBuffer
        filename: str,
        render: bool = False
    ):
    # Environment setup
    env = Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    buffer = buffer_class(BUFFER_SIZE)
    q_network = q_network_class(state_size=env.state_size, action_size=env.action_size)
    agent = DQN(
        q_network=q_network,
        buffer=buffer,
        action_size=env.action_size,
        gamma=0.99,
        lr=LR,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        tau=TAU,
        device=device
    )

    if render: 
        renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, title, env)
        renderer.window_init()

    # Training tracking
    last_100_rewards = deque(maxlen=100)
    last_100_success = deque(maxlen=100)
    last_100_times = deque(maxlen=100)
    last_100_rtimes = deque(maxlen=100)
    last_100_steps = deque(maxlen=100)
    total_training_time = 0.0
    epsilon = EPSILON_START

    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        "episode", "avg_rewards", "avg_steps", "avg_success",
        "epsilon", "time_elapsed", "real_time_elapsed", "avg_loss"
    ])

    episode_losses = []  # Track losses per episode

    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        real_time_elapsed = 0.0
        state = env.reset()
        done = False
        start_time = time.perf_counter()

        while not done:
            # Get action and step
            action = agent.act(state)
            next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
            
            # Store experience and train
            agent.buffer.add((state, action, reward, next_state, done))
            loss = agent.update(BATCH_SIZE)
            
            if loss is not None:
                episode_losses.append(loss)  # <-- Collect batch losses

            state = next_state
            real_time_elapsed += PHYSICS_DT

            if render:
                renderer.render()

        episode_reward = env.episode_reward
        episode_steps = env.episode_steps
        success = env.success
        
        # Update tracking
        time_elapsed = time.perf_counter() - start_time
        total_training_time += time_elapsed
        last_100_rewards.append(episode_reward)
        last_100_success.append(1 if success else 0)
        last_100_times.append(time_elapsed)
        last_100_rtimes.append(real_time_elapsed)
        last_100_steps.append(episode_steps)

        # Calculate averages
        avg_rewards = np.mean(last_100_rewards) if episode >= 100 else np.nan
        avg_success = np.mean(last_100_success) * 100 if episode >= 100 else np.nan
        avg_time = np.mean(last_100_times) if episode >= 100 else np.nan
        avg_rtime = np.mean(last_100_rtimes) if episode >= 100 else np.nan
        avg_steps = np.mean(last_100_steps) if episode >= 100 else np.nan

        # Print every episode (flushing)
        print(
            f"Type: {title} | "
            f"Ep {episode:>6d} | "
            f"S:{episode_steps:>4d} | "
            f"R:{episode_reward:>7.2f} | "
            f"ε:{epsilon:>.3f} | "
            f"T:{time_elapsed:>5.2f}s | "
            f"RT:{real_time_elapsed:>5.2f}s",
            end='\r',
            flush=True
        )

        real_time_elapsed = 0.0

        avg_loss = np.mean(episode_losses) if episode_losses else np.nan  # <-- New metric

        # Every 100 episodes
        if episode % 100 == 0:
            # Detailed non-flushing print
            print("\n" + "="*80)
            print(f"- Last 100 Avg Time: {avg_time:.2f}s per episode")
            print(f"- Last 100 Avg Real Time: {avg_rtime:.2f}s per episode")
            print(f"- Last 100 Avg Steps: {avg_steps:.2f}")
            print(f"- Last 100 Avg Reward: {avg_rewards:.2f}")
            print(f"- Last 100 Success Rate: {avg_success:.2f}%")
            print(f"- Total Training Time: {total_training_time:.2f}s")
            print(f"- Current Epsilon: {epsilon:.4f}")
            print(f"- Avg Loss: {avg_loss:.4f}")
            print("="*80 + "\n")

            # Add to DataFrame
            training_stats.loc[len(training_stats)] = {
                "episode": episode,
                "avg_steps": avg_steps,
                "avg_rewards": avg_rewards,
                "avg_success": avg_success,
                "epsilon": epsilon,
                "time_elapsed": time_elapsed,
                "real_time_elapsed": real_time_elapsed,
                "avg_loss": avg_loss
            }

        # Epsilon decay every episode (fixed placement)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Final save and plot
    csv_dir = "csv_data"
    training_stats.to_csv(f'{csv_dir}/{filename}.csv', index=False)

    if render: 
        renderer.close_window()


def train_jump(
        title: str,
        q_network_class: Type[nn.Module],  # Your QNetwork class (Shallow/Medium/Deep)
        buffer_class: Buffer,  # Your ReplayBuffer
        filename: str,
        render: bool = False
    ):
    # Environment setup
    env = Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    buffer = buffer_class(BUFFER_SIZE)
    q_network = q_network_class(state_size=env.state_size, action_size=env.action_size)
    agent = DQN(
        q_network=q_network,
        buffer=buffer,
        action_size=env.action_size,
        gamma=0.99,
        lr=LR,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        tau=TAU,
        device=device
    )

    if render: 
        renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, title, env)
        renderer.window_init()

    # Training tracking
    last_100_rewards = deque(maxlen=100)
    last_100_success = deque(maxlen=100)
    last_100_times = deque(maxlen=100)
    last_100_rtimes = deque(maxlen=100)
    last_100_steps = deque(maxlen=100)
    total_training_time = 0.0
    epsilon = EPSILON_START

    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        "episode", "avg_rewards", "avg_steps", "avg_success",
        "epsilon", "time_elapsed", "real_time_elapsed", "avg_loss"
    ])

    episode_losses = []  # Track losses per episode

    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        real_time_elapsed = 0.0
        state = env.reset()
        done = False
        start_time = time.perf_counter()

        reward_accumulator = 0
        release_state = None

        while not done:
            # Get action and step
            action = agent.act(state)
            if action == 4:
                release_state = state
                while not done:
                    next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
                    reward_accumulator += reward
                    if render:
                        renderer.render()
                    real_time_elapsed += PHYSICS_DT
                reward = reward_accumulator
                state = release_state
                action = 4
            else:
                next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
                if render:
                    renderer.render()
            
            # Store experience and train
            agent.buffer.add((state, action, reward, next_state, done))
            loss = agent.update(BATCH_SIZE)
            
            if loss is not None:
                episode_losses.append(loss)  # <-- Collect batch losses

            state = next_state
            real_time_elapsed += PHYSICS_DT

        episode_reward = env.episode_reward
        episode_steps = env.episode_steps
        success = env.success
        
        # Update tracking
        time_elapsed = time.perf_counter() - start_time
        total_training_time += time_elapsed
        last_100_rewards.append(episode_reward)
        last_100_success.append(1 if success else 0)
        last_100_times.append(time_elapsed)
        last_100_rtimes.append(real_time_elapsed)
        last_100_steps.append(episode_steps)

        # Calculate averages
        avg_rewards = np.mean(last_100_rewards) if episode >= 100 else np.nan
        avg_success = np.mean(last_100_success) * 100 if episode >= 100 else np.nan
        avg_time = np.mean(last_100_times) if episode >= 100 else np.nan
        avg_rtime = np.mean(last_100_rtimes) if episode >= 100 else np.nan
        avg_steps = np.mean(last_100_steps) if episode >= 100 else np.nan

        # Print every episode (flushing)
        print(
            f"Type: {title} | "
            f"Ep {episode:>6d} | "
            f"S:{episode_steps:>4d} | "
            f"R:{episode_reward:>7.2f} | "
            f"ε:{epsilon:>.3f} | "
            f"T:{time_elapsed:>5.2f}s | "
            f"RT:{real_time_elapsed:>5.2f}s",
            end='\r',
            flush=True
        )

        real_time_elapsed = 0.0

        avg_loss = np.mean(episode_losses) if episode_losses else np.nan  # <-- New metric

        # Every 100 episodes
        if episode % 100 == 0:
            # Detailed non-flushing print
            print("\n" + "="*80)
            print(f"- Last 100 Avg Time: {avg_time:.2f}s per episode")
            print(f"- Last 100 Avg Real Time: {avg_rtime:.2f}s per episode")
            print(f"- Last 100 Avg Steps: {avg_steps:.2f}")
            print(f"- Last 100 Avg Reward: {avg_rewards:.2f}")
            print(f"- Last 100 Success Rate: {avg_success:.2f}%")
            print(f"- Total Training Time: {total_training_time:.2f}s")
            print(f"- Current Epsilon: {epsilon:.4f}")
            print(f"- Avg Loss: {avg_loss:.4f}")
            print("="*80 + "\n")

            # Add to DataFrame
            training_stats.loc[len(training_stats)] = {
                "episode": episode,
                "avg_steps": avg_steps,
                "avg_rewards": avg_rewards,
                "avg_success": avg_success,
                "epsilon": epsilon,
                "time_elapsed": time_elapsed,
                "real_time_elapsed": real_time_elapsed,
                "avg_loss": avg_loss
            }

        # Epsilon decay every episode (fixed placement)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Final save and plot
    csv_dir = "csv_data"
    training_stats.to_csv(f'{csv_dir}/{filename}.csv', index=False)

    if render: 
        renderer.close_window()



