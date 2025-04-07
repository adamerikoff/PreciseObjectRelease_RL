import random
import pyray as pr
import math
import environment
import time

def main():
    pr.init_window(1200, 800, "Drone Grenade Environment")
    pr.set_target_fps(60)

    env = environment.Environment((500, 400, 500))
    obs = env.reset()

    # Physics timing variables
    physics_dt = 1.0/60.0  # Fixed physics timestep (60Hz)
    accumulator = 0.0
    last_time = time.perf_counter()
    
    # Real-time measurement variables
    episode_start_real_time = time.perf_counter()
    total_real_time = 0.0

    while not pr.window_should_close():
        # Calculate delta time
        current_time = time.perf_counter()
        frame_time = current_time - last_time
        last_time = current_time

        # Prevent spiral of death on slow systems
        frame_time = min(frame_time, 0.25)

        # Accumulate physics time
        accumulator += frame_time

        # Process physics in fixed timesteps
        while accumulator >= physics_dt:
            action = random.randint(0, 4)
            obs, reward, done, info = env.step(action, physics_dt)
            accumulator -= physics_dt

            if done:
                episode_end_real_time = time.perf_counter()
                real_time_elapsed = episode_end_real_time - episode_start_real_time
                total_real_time += real_time_elapsed
                
                print("═" * 60)
                print(f"🔥 COLLISION X: {env.grenade.pos.x:>10.2f} | Z: {env.grenade.pos.z:>7.2f}")
                print(f"⏱️ SIM TIME: {env.episode_time:>10.2f}s | THEORETICAL: {env.theoretical_time_required:>7.2f}s")
                print(f"🕒 REAL TIME: {real_time_elapsed:.4f}s (Wall Clock) | {env.episode_time/real_time_elapsed:.2f}x Speed")
                print(f"Σ TOTAL REAL TIME: {total_real_time:.2f}s")
                print("═" * 60)
                current_real_time = time.perf_counter() - episode_start_real_time
                print(f"Real Time: {current_real_time:.2f}s", 10, 40, 20, pr.BLACK)
                print(f"Sim Time: {env.episode_time:.2f}s", 10, 70, 20, pr.BLACK)
                print(f"Speed: {env.episode_time/max(0.001, current_real_time):.1f}x", 10, 100, 20, pr.BLACK)
                obs = env.reset()
                episode_start_real_time = time.perf_counter()

        # Rendering
        env.render()

    pr.close_window()

if __name__ == "__main__":
    main()