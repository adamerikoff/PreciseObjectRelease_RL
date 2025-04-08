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
    PHYSICS_DT = 1.0 / TARGET_FPS
    SIMULATION_SPEED = 100.0
    N_EPISODES = 50000
    
    # Training parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # Initialize window and environment
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)
    pr.set_target_fps(TARGET_FPS)
    
    env = environment.Environment((500, 400, 500))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.state_size, env.action_size, seed=0, device=device)
    
    # Training tracking
    last_100_rewards = deque(maxlen=100)  # For 100-episode average
    last_10_rewards = deque(maxlen=10)    # For 10-episode average
    epsilon = EPSILON_START
    total_real_time = 0.0
    total_episode_steps = 0

    def print_episode(episode, steps, reward, avg10, eps, time_elapsed):
        """Print single-line episode summary"""
        print(
            f"Ep {episode:>6d} | "
            f"S:{steps:>4d} | "
            f"R:{reward:>7.2f} | "
            f"Avg10:{avg10:>7.2f} | "
            f"ε:{eps:>.3f} | "
            f"T:{time_elapsed:>5.2f}s", 
            flush=True
        )

    def print_100ep_summary(episode, avg100, total_time, eps):
        """Print detailed 100-episode summary"""
        print("\n" + "="*80)
        print(f"EPISODE {episode} SUMMARY:")
        print(f"- Last 100 Avg Reward: {avg100:.2f}")
        print(f"- Total Training Time: {total_time/3600:.2f} hours")
        print(f"- Current Epsilon: {eps:.4f}")
        print(f"- Total Steps: {total_episode_steps}")
        print("="*80 + "\n")

    # Main training loop
    for episode in range(1, N_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        start_time = time.perf_counter()
        
        while not done and not pr.window_should_close():
            # Time management
            current_time = time.perf_counter()
            frame_time = min(current_time - start_time, 0.25) * SIMULATION_SPEED
            accumulator = frame_time
            
            # Physics steps
            while accumulator >= PHYSICS_DT and not done:
                state_array = np.array(state, dtype=np.float32)
                action = agent.act(state_array, epsilon)
                next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT)
                
                episode_reward += reward
                episode_steps += 1
                agent.step(state_array, action, reward, next_state, done)
                state = next_state
                accumulator -= PHYSICS_DT

        # Update tracking
        time_elapsed = time.perf_counter() - start_time
        total_real_time += time_elapsed
        total_episode_steps += episode_steps
        last_100_rewards.append(episode_reward)
        last_10_rewards.append(episode_reward)
        
        # Calculate averages
        avg10 = np.mean(last_10_rewards) if last_10_rewards else 0
        avg100 = np.mean(last_100_rewards) if last_100_rewards else 0
        
        # Print every episode
        print_episode(
            episode, episode_steps, episode_reward, 
            avg10, epsilon, time_elapsed
        )
        
        # Print detailed summary every 100 episodes
        if episode % 100 == 0:
            print_100ep_summary(episode, avg100, total_real_time, epsilon)
        
        if episode % 10 == 0:
            # Epsilon decay every episode
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    pr.close_window()


if __name__ == "__main__":
    main()