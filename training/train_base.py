from collections import deque
from typing import Tuple
import time
import random 

import torch
import torch.nn as nn
import numpy as np

from CONFIG import GAMMA, LR, BUFFER_SIZE, BATCH_SIZE, UPDATE_INTERVAL, SCREEN_WIDTH, SCREEN_HEIGHT, EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY, N_EPISODES, PHYSICS_DT, TAU, RENDERING_SLEEP, HEIGHTS
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

    # Initialize DataFrame for tracking
    training_stats = pd.DataFrame(columns=[
        "episode", "height", "reward", "steps", 
        "success", "epsilon", "actions_forward",
        "actions_backward", "actions_left", "actions_right", 
        "wind_vector", "terminal_distance"
    ])
    
    last_100 = {
        'rewards': deque(maxlen=100),
        'success': deque(maxlen=100),
        'steps': deque(maxlen=100),
        'actions': {a: deque(maxlen=100) for a in ["forward", "backward", "left", "right"]}
    }

    total_training_time = 0.0
    total_episodes = 0

    for base_height in HEIGHTS:
        epsilon = EPSILON_MAX
        for episode in range(1, N_EPISODES + 1):
            total_episodes += 1
            height = random.uniform(base_height - 50.0, base_height + 50.0)

            # Run episode
            state = env.reset(height=height)
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
            
            # Update tracking
            time_elapsed = time.perf_counter() - start_time
            total_training_time += time_elapsed

            last_100['rewards'].append(env.episode_reward)
            last_100['success'].append(int(env.success))
            last_100['steps'].append(env.episode_steps)
            for act in last_100['actions']:
                last_100['actions'][act].append(env.action_count[act])

            # Epsilon decay
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            # Store episode data
            training_stats.loc[len(training_stats)] = {
                "episode": total_episodes,
                "height": height,
                "reward": env.episode_reward,
                "steps": env.episode_steps,
                "success": env.success,
                "epsilon": epsilon,
                "actions_forward": env.action_count["forward"],
                "actions_backward": env.action_count["backward"],
                "actions_left": env.action_count["left"],
                "actions_right": env.action_count["right"],
                "wind_vector": env.wind.tolist(),
                "terminal_distance": env.terminal_distance
            }

            # Print every episode (flushing)
            print(
                f"Type: {title} | "
                f"Ep {total_episodes:>7d} | "
                f"H {height:>7.3f} | "
                f"S:{env.episode_steps:>7d} | "
                f"R:{env.episode_reward:>7.3f} | "
                f"ε:{epsilon:>7.3f} | "
                f"T:{time_elapsed:>7.3f}s | "
                f"RT:{ env.episode_time:>7.3f}s",
                end='\r',
                flush=True
            )

            # Print progress every 100 episodes or height change
            if total_episodes % 100 == 0:
                avg_reward = np.mean(last_100['rewards']) if total_episodes >= 100 else np.nan
                success_rate = np.mean(last_100['success']) * 100 if total_episodes >= 100 else np.nan
                
                print(
                    f"BH:{base_height:>5}m | H:{height:>7.3f} | Ep:{total_episodes:5}/{len(HEIGHTS) * N_EPISODES} | "
                    f"R:{avg_reward:7.2f} | S:{success_rate:5.1f}% | "
                    f"ε:{epsilon:.3f}"
                )

        # Add to DataFrame
        agent.save(f"models/{filename}_checkpoint_{base_height}.pt")
    # Final save and plot
    csv_dir = "csv_data"
    training_stats.to_csv(f'{csv_dir}/{filename}.csv', index=False)

    if render: 
        renderer.close_window()
    