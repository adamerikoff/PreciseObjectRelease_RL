import time
import pyray as pr
import environment
from dqn import DQNAgent
import numpy as np
import torch
from collections import deque


def main():
    # Constants
    SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
    WINDOW_TITLE = "Drone Grenade Environment"
    TARGET_FPS = 200
    PHYSICS_DT = 1.0 / TARGET_FPS  # Fixed physics timestep
    SIMULATION_SPEED = 20.0  # Adjust simulation speed
    N_EPISODES = 500000
    
    # Training parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # Initialize window and environment
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)
    pr.set_target_fps(TARGET_FPS)
    
    env = environment.Environment((500, 400, 500))
    state_size = env.state_size
    action_size = env.action_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize DQN Agent
    agent = DQNAgent(state_size, action_size, seed=0, device=device)
    
    # Training tracking
    episode_rewards = deque(maxlen=100)  # Track last 100 episodes
    epsilon = EPSILON_START
    total_real_time = 0.0
    
    def print_progress(episode, episode_steps, episode_reward, avg_reward, epsilon, real_time):
        """Print formatted training progress information."""
        progress = (
            f"Ep: {episode:>6}/{N_EPISODES} | "
            f"Steps: {episode_steps:>4} | "
            f"Reward: {episode_reward:>7.2f} | "
            f"Avg100: {avg_reward:>7.2f} | "
            f"Eps: {epsilon:>5.3f} | "
            f"Time: {real_time:>5.2f}s"
        )
        print(f"\r{progress}", end="", flush=True)
    
    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        episode_start = time.perf_counter()
        accumulator = 0.0
        last_time = time.perf_counter()
        
        while not done and not pr.window_should_close():
            # Time management
            current_time = time.perf_counter()
            frame_time = min(current_time - last_time, 0.25) * SIMULATION_SPEED
            last_time = current_time
            accumulator += frame_time
            
            # Physics steps
            while accumulator >= PHYSICS_DT and not done:
                # Agent action
                state_array = np.array(state, dtype=np.float32)
                action = agent.act(state_array, epsilon)
                
                # Environment step
                next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
                
                # Update tracking
                episode_reward += reward
                if state[-1] == 0:  # Only count steps before release?
                    episode_steps += 1
                
                # Agent learning
                agent.step(state_array, action, reward, next_state, done)
                state = next_state
                accumulator -= PHYSICS_DT
            
            # Rendering
            env.render()
            
            # Episode completion handling
            if done:
                episode_rewards.append(episode_reward)
                real_time_elapsed = time.perf_counter() - episode_start
                total_real_time += real_time_elapsed
                
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                print_progress(episode, episode_steps, episode_reward, avg_reward, epsilon, real_time_elapsed)
                
                # Periodic updates
                if episode % 10 == 0:
                    epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
                    episode_steps = 0
                    print()  # New line for better readability
                
                if episode % 1000 == 0:
                    print(f"\n=== Episode {episode} Summary ===")
                    print(f"Total training time: {total_real_time/3600:.2f} hours")
                    print(f"Current epsilon: {epsilon:.4f}")
                    print(f"Average reward (last 100): {avg_reward:.2f}\n")
                
                break
    
    pr.close_window()


if __name__ == "__main__":
    main()