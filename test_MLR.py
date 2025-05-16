import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pyray as pr
from typing import Optional, Tuple
from src.environment import Environment
from src.renderer import Renderer
from CONFIG import SCENE_X, SCENE_Z, SCENE_Y, SCREEN_HEIGHT, SCREEN_WIDTH, PHYSICS_DT, RENDERING_SLEEP

class ImprovedLandingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            
            nn.Linear(32, 2)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        return self.model(x)

def calculate_optimal_action(
    drone_pos: np.ndarray, 
    target_pos: np.ndarray, 
    predicted_landing: np.ndarray,
    drone_speed: float,
    current_wind: np.ndarray,
    model: nn.Module
) -> int:
    current_error = np.sum((target_pos[[0,2]] - predicted_landing)**2)
    actions = [0, 1, 2, 3]  # right, left, back, forward
    best_action = None
    min_error = current_error
    
    for action in actions:
        new_drone_pos = drone_pos.copy()
        if action == 0:
            new_drone_pos[0] += drone_speed
        elif action == 1:
            new_drone_pos[0] -= drone_speed
        elif action == 2:
            new_drone_pos[2] -= drone_speed
        elif action == 3:
            new_drone_pos[2] += drone_speed
            
        model_input = torch.FloatTensor([
            new_drone_pos[0], new_drone_pos[1], new_drone_pos[2],
            current_wind[0], current_wind[2]
        ]).unsqueeze(0)
        
        with torch.no_grad():
            new_prediction = model(model_input).numpy()[0]
        
        new_error = np.sum((target_pos[[0,2]] - new_prediction)**2)
        
        if new_error < min_error:
            min_error = new_error
            best_action = action
    
    return best_action if min_error < current_error else None

def run_episode(env: Environment, renderer, model: nn.Module, auto_release: bool = False) -> Tuple[bool, float]:
    done = False
    success = False
    total_reward = 0.0
    release_counter = 0
    max_positioning_steps = 100  # Max steps before auto-release
    
    while not done and not pr.window_should_close():
        current_drone_pos = env.drone.position
        current_wind = env.wind
        current_target_pos = env.target.position

        model_input = torch.FloatTensor([
            current_drone_pos[0], current_drone_pos[1], current_drone_pos[2],
            current_wind[0], current_wind[2]
        ]).unsqueeze(0)
        
        with torch.no_grad():
            predicted_landing = model(model_input).numpy()[0]

        action = calculate_optimal_action(
            current_drone_pos, current_target_pos, 
            predicted_landing, env.drone.speed, 
            current_wind, model
        )

        # Auto-release logic
        if auto_release and release_counter >= max_positioning_steps:
            action = 4  # Release ball
            release_counter = 0
        
        obs, reward, done = env.step(action, PHYSICS_DT)
        total_reward += reward
        release_counter += 1

        # renderer.render(predicted_landing)

        if done:
            success = reward > 0  # Assuming positive reward means success
            break

    return success, total_reward

def run_test_suite(num_episodes: int = 100) -> pd.DataFrame:
    # Initialize model
    model = ImprovedLandingPredictor()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Initialize environment and renderer
    env = Environment(np.array([SCENE_X, SCENE_Y, SCENE_Z]))
    renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, "Auto-Testing Mode", env, True, True, RENDERING_SLEEP)
    renderer.window_init()
    
    # Results tracking
    results = []
    
    for episode in range(1, num_episodes + 1):
        env.reset()
        success, reward = run_episode(env, renderer, model, auto_release=True)
        
        results.append({
            'episode': episode,
            'success': success,
            'reward': reward,
            'distance': env.terminal_distance,
            'height': env.drone.position[1],
            'wind': env.wind
        })
        
        print(f"Episode {episode}/{num_episodes} - {'Success' if success else 'Failure'} - Reward: {reward:.2f}")
    
    renderer.close_window()
    
    # Create and analyze DataFrame
    df = pd.DataFrame(results)
    df['cumulative_success'] = df['success'].cumsum()
    df['success_rate'] = df['cumulative_success'] / (df.index + 1)
    
    # Save results
    df.to_csv('drone_test_results.csv', index=False)
    print("\nTest Summary:")
    print(f"Total Successes: {df['success'].sum()}/{num_episodes}")
    print(f"Final Success Rate: {df['success_rate'].iloc[-1]:.2%}")
    
    return df

if __name__ == "__main__":
    # Run the test suite
    results_df = run_test_suite(num_episodes=1000)
    
    # Optionally show the results
    print("\nFirst 10 episodes:")
    print(results_df.head(10))
    
    print("\nLast 10 episodes:")
    print(results_df.tail(10))