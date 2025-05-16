from collections import deque
from typing import Tuple
import time
import random 

import torch
import torch.nn as nn
import numpy as np

from CONFIG import SCREEN_WIDTH, SCREEN_HEIGHT, EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY, N_EPISODES, PHYSICS_DT, TAU, RENDERING_SLEEP, HEIGHTS
from src.environment import Environment
from src.renderer import Renderer
from agent.dqn import DQNAgent, QNetwork, ReplayBuffer


import pandas as pd 
import numpy as np

def collect_data(
        title: str,
        filename: str,
        render: bool = False,
        top: bool = False,
        episode_n: int = 5000
    ):

    env = Environment(np.array([500, 400, 500]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if render: 
        renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, title, env, top, True, RENDERING_SLEEP)
        renderer.window_init()

    # Initialize DataFrame for tracking
    data = pd.DataFrame(columns=[
        "episode", "i_x", "i_y", "i_z", 
        "f_x", "f_y", "f_z", "w_x", "w_z"
    ])
    for episode in range(1, episode_n + 1):
        height = random.uniform(50, 400)

        # Run episode
        state = env.reset(height=height)
        done = False

        initial_position = env.drone.position
        while not done:
            action = 4
            next_state, reward, done = env.step(action, PHYSICS_DT)
            if render: renderer.render()
        final_position = env.ball.position

        # Store episode data
        data.loc[len(data)] = {
            "episode": episode,
            "i_x": initial_position[0], 
            "i_y": initial_position[1], 
            "i_z": initial_position[2], 
            "f_i": final_position[0], 
            "f_y": final_position[1], 
            "f_z": final_position[2],
            "w_i": env.wind[0], "w_z": env.wind[2]
        }

        # Print every episode (flushing)
        print(
            f"Type: {title} | "
            f"Ep {episode:>7d}/{episode_n} | ",
            end='\r',
            flush=True
        )

    # Final save and plot
    data_dir = "data"
    data.to_csv(f'{data_dir}/{filename}.csv', index=False)

    if render: 
        renderer.close_window()
    

collect_data("DATA_COLLECTION", "data_random", False, False, 500_000)