import random
import time
import pyray as pr
import environment


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

    # Physics timing setup
    PHYSICS_DT = 1.0 / TARGET_FPS  # Fixed physics timestep
    accumulator = 0.0
    last_time = time.perf_counter()
    
    # Real-time measurement setup
    episode_start_real_time = last_time
    total_real_time = 0.0

    # Simulation speed control (1.0 = realtime, 2.0 = 2x speed, etc.)
    SIMULATION_SPEED = 5.0  # Adjust this value to change simulation speed

    # Main game loop
    while not pr.window_should_close():
        # Time management
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        last_time = current_time

        # Prevent spiral of death on slow systems
        frame_time = min(frame_time, 0.25)

        # Apply simulation speed multiplier
        frame_time *= SIMULATION_SPEED

        # Accumulate physics time
        accumulator += frame_time

        # Process physics in fixed timesteps
        while accumulator >= PHYSICS_DT:
            # Take random action (replace with your control logic)
            action = random.randint(0, 4)
            obs, reward, done, info = env.step(action, PHYSICS_DT)
            accumulator -= PHYSICS_DT

            if done:
                # Calculate timing statistics
                episode_end_real_time = time.perf_counter()
                real_time_elapsed = episode_end_real_time - episode_start_real_time
                total_real_time += real_time_elapsed
                
                # Print episode summary
                print_episode_summary(env, real_time_elapsed, total_real_time)
                
                # Reset environment
                obs = env.reset()
                episode_start_real_time = time.perf_counter()

        # Rendering
        env.render()

    pr.close_window()


def print_episode_summary(env, real_time_elapsed, total_real_time):
    speed_ratio = env.episode_time / real_time_elapsed
    
    collision_x = f"{env.grenade.pos.x:>10.2f}"
    collision_z = f"{env.grenade.pos.z:>7.2f}"
    sim_time = f"{env.episode_time:>10.2f}"
    theory_time = f"{env.theoretical_time_required:>7.2f}"
    real_time = f"{real_time_elapsed:>7.4f}"
    speed = f"{speed_ratio:>5.2f}"
    total_time = f"{total_real_time:>7.2f}"

    separator = "═" * 60
    output = [
        separator,
        f"🔥 COLLISION  X: {collision_x}  |  Z: {collision_z}",
        f"⏱️ SIM TIME: {sim_time}s  |  THEORETICAL: {theory_time}s",
        f"🕒 REAL TIME: {real_time}s (Wall Clock)  |  {speed}x Speed",
        f"Σ TOTAL REAL TIME: {total_time}s",
        separator
    ]

    print("\n".join(output))


if __name__ == "__main__":
    main()