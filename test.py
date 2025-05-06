import os
import torch
import pandas as pd
import numpy as np

from src.environment import Environment
from src.renderer import Renderer
from agent.dqn import DQNAgent
from CONFIG import PHYSICS_DT, SCREEN_HEIGHT, SCREEN_WIDTH, SCENE_Y, SCENE_X, SCENE_Z, RENDERING_SLEEP

def test_model_across_heights(model_path, num_sets=5, episodes_per_set=50):
    # Initialize environment and agent
    env = Environment(np.array([SCENE_X, SCENE_Y, SCENE_Z]))
    renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, model_path, env, False, True, RENDERING_SLEEP, False)

    # renderer.window_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_size, device=device)
    
    # Load trained model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    test_heights = [50, 100, 150, 200, 250, 300, 350, 400]
    
    # Results dataframe
    results = pd.DataFrame(columns=[
        'set_num', 'height', 'episode', 'reward', 
        'steps', 'success', 'actions_forward',
        'actions_backward', 'actions_left', 'actions_right',
        'wind_vector', 'terminal_distance'
    ])

    # Fixed epsilon for evaluation (small exploration)
    eval_epsilon = 0.01

    for height in test_heights:
        for set_num in range(1, num_sets + 1):
            for episode in range(1, episodes_per_set + 1):
                # Run episode with current height
                state = env.reset(height=height)
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.act(state, eval_epsilon)
                    next_state, reward, done = env.step(action, PHYSICS_DT)
                    state = next_state
                    episode_reward += reward
                    # renderer.render()
                
                # Record results
                results.loc[len(results)] = {
                    'set_num': set_num,
                    'height': height,
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': env.episode_steps,
                    'success': int(env.success),
                    'actions_forward': env.action_count["forward"],
                    'actions_backward': env.action_count["backward"],
                    'actions_left': env.action_count["left"],
                    'actions_right': env.action_count["right"],
                    'wind_vector': env.wind.tolist(),
                    'terminal_distance': env.terminal_distance
                }
                
            # Print progress
            print(f"Set {set_num}/{num_sets} | Height {height}m | "
                  f"Success Rate: {results[results['height'] == height]['success'].mean()*100:.1f}%")

    # renderer.close_window()

    return results



folder_name = "models"
models = [f for f in os.listdir(folder_name) if f.endswith('.pt')]
for model in models:
    results_df = test_model_across_heights(f"{folder_name}/{model}")
    csv_path = f"csv_data/test_data/{model}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Testing complete. Results saved to {csv_path}.")
