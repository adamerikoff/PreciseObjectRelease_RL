import time
import pyray as pr
import environment
from dqn import DQNAgent
import numpy as np
import torch


def main():
    # Initialize window and environment
    SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
    WINDOW_TITLE = "Drone Grenade Environment"
    TARGET_FPS = 100

    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)
    pr.set_target_fps(TARGET_FPS)

    # Environment setup
    env = environment.Environment((500, 400, 500))
    obs = env.reset()
    state_size = env.state_size  # Get the size of your state vector
    action_size = env.action_size  # Assuming your environment has an action_space_n attribute
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DQN Agent
    agent = DQNAgent(state_size, action_size, seed, device=device)

    # Hyperparameters for training
    n_episodes = 500000  # Adjust as needed
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    episode_rewards = []
    update_every = agent.update_every

    # Physics timing setup
    PHYSICS_DT = 1.0 / TARGET_FPS  # Fixed physics timestep
    accumulator = 0.0
    last_time = time.perf_counter()

    # Real-time measurement setup
    episode_start_real_time = last_time
    total_real_time = 0.0
    total_steps = 0

    # Simulation speed control (1.0 = realtime, 2.0 = 2x speed, etc.)
    SIMULATION_SPEED = 5.0  # Adjust this value to change simulation speed

    # Main game loop
    for episode in range(1, n_episodes + 1):
        episode_reward = 0
        done = False
        state = env.reset()
        episode_start_real_time = time.perf_counter()

        while not done and not pr.window_should_close():
            # Time management
            current_time = time.perf_counter()
            frame_time = current_time - last_time
            last_time = current_time
            frame_time = min(frame_time, 0.25)
            frame_time *= SIMULATION_SPEED
            accumulator += frame_time

            while accumulator >= PHYSICS_DT:
                # Agent takes action
                state = np.array(state, dtype=np.float32)  # Ensure state is a NumPy array
                action = agent.act(state, epsilon)

                # Environment step
                next_state, reward, done = env.step(env.action_space[action], PHYSICS_DT) 
                
                episode_reward += reward
                total_steps += 1
                accumulator -= PHYSICS_DT

                # Agent learns
                agent.step(state, action, reward, next_state, done)
                state = next_state

                if done:
                    episode_end_real_time = time.perf_counter()
                    real_time_elapsed = episode_end_real_time - episode_start_real_time
                    total_real_time += real_time_elapsed
                    episode_rewards.append(episode_reward)
                    

                    avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    print(f"\rEpisode {episode}/{n_episodes}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg Reward (Last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}, Real Time: {real_time_elapsed:.2f}", end="")
                    if episode % 100 == 0:
                        print(f"\rEpisode {episode}/{n_episodes}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg Reward (Last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}, Real Time: {real_time_elapsed:.2f}")
                    break

            # Rendering
            env.render()

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

    pr.close_window()