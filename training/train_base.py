from collections import deque
from typing import Tuple
import time

import torch
import torch.nn as nn
import numpy as np

from CONFIG import GAMMA, LR, BUFFER_SIZE, BATCH_SIZE, UPDATE_INTERVAL, SCREEN_WIDTH, SCREEN_HEIGHT, EPSILON_START, EPSILON_END, EPSILON_DECAY, N_EPISODES, PHYSICS_DT, TAU, RENDERING_SLEEP
from src.environment import Environment
from src.renderer import Renderer
from agent.dqn import DQNAgent, QNetwork, ReplayBuffer


import pandas as pd 
import numpy as np

def train(
        title: str,
        filename: str,
        render: bool = False,
        jump: bool = False,
        top: bool = False
    ):

    env = Environment(np.array([500, 400, 500]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(env.state_size, env.action_size, GAMMA, LR, BUFFER_SIZE, BATCH_SIZE, UPDATE_INTERVAL, TAU, device)

    if render: 
        renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, title, env, top, True, RENDERING_SLEEP)
        renderer.window_init()

    # Training tracking
    last_100_rewards = deque(maxlen=100)
    last_100_success = deque(maxlen=100)
    last_100_times = deque(maxlen=100)
    last_100_rtimes = deque(maxlen=100)
    last_100_steps = deque(maxlen=100)

    last_100_action_forward = deque(maxlen=100)
    last_100_action_backward = deque(maxlen=100)
    last_100_action_left = deque(maxlen=100)
    last_100_action_right = deque(maxlen=100)
    last_100_action_release = deque(maxlen=100)

    total_training_time = 0.0
    epsilon = EPSILON_START

    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        "episode", "avg_rewards", "avg_steps", "avg_success",
        "epsilon", "time_elapsed", "real_time_elapsed"
    ])

    epsilons = []

    # Main training loop
    for episode in range(1, N_EPISODES + 1):

        real_time_elapsed = 0.0
        
        state = env.reset()
        done = False
        start_time = time.perf_counter()

        if jump:
            while not done:
                action = agent.act(state, epsilon)
                if action == 4:
                    reward_accumulator = 0.0
                    release_state = state
                    while not env.check_done():
                        next_state, reward, done = env.step(action, PHYSICS_DT)
                        reward_accumulator += reward
                        if render: renderer.render()
                    reward = reward_accumulator
                    state = release_state
                else:
                    next_state, reward, done = env.step(action, PHYSICS_DT)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                if render: renderer.render()
        else:
            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done = env.step(action, PHYSICS_DT)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                if render: renderer.render()

        real_time_elapsed = env.episode_time
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
        last_100_action_forward.append(env.action_count["forward"])
        last_100_action_backward.append(env.action_count["backward"])
        last_100_action_left.append(env.action_count["left"])
        last_100_action_right.append(env.action_count["right"])
        last_100_action_release.append(env.action_count["release"])

        # Calculate averages
        avg_rewards = np.mean(last_100_rewards) if episode >= 100 else np.nan
        avg_success = np.mean(last_100_success) * 100 if episode >= 100 else np.nan
        avg_time = np.mean(last_100_times) if episode >= 100 else np.nan
        avg_rtime = np.mean(last_100_rtimes) if episode >= 100 else np.nan
        avg_steps = np.mean(last_100_steps) if episode >= 100 else np.nan
        avg_action_forward = np.mean(last_100_action_forward) if episode >= 100 else np.nan
        avg_action_backward = np.mean(last_100_action_backward) if episode >= 100 else np.nan
        avg_action_left = np.mean(last_100_action_left) if episode >= 100 else np.nan
        avg_action_right = np.mean(last_100_action_right) if episode >= 100 else np.nan
        avg_action_release = np.mean(last_100_action_release) if episode >= 100 else np.nan

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
            print(f"- Actions: f={avg_action_forward:.2f} b={avg_action_backward:.2f} l={avg_action_left:.2f} r={avg_action_right:.2f} drop={avg_action_release:.2f}")
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
            }

        # Epsilon decay every episode (fixed placement)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Final save and plot
    csv_dir = "csv_data"
    training_stats.to_csv(f'{csv_dir}/{filename}.csv', index=False)

    if render: 
        renderer.close_window()
    