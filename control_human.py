import time
import pyray as pr
import environment


def main():
    # Initialize window and environment
    SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
    WINDOW_TITLE = "Drone Grenade Environment"
    
    pr.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE)

    PHYSICS_DT = 0.05
    PHI = 0

    # Environment setup
    env = environment.Environment((500, 400, 500))
    obs = env.reset(phi=PHI)
    # Physics timing setup
    
    
    # Real-time measurement setup
    start_time = time.perf_counter()
    sim_time = 0.0

    done = False

    # Main game loop
    while not done and not pr.window_should_close():
        # Take random action (replace with your control logic)
        action = None
        if pr.is_key_down(pr.KEY_UP): action = "forward"
        if pr.is_key_down(pr.KEY_DOWN): action = "backward"
        if pr.is_key_down(pr.KEY_RIGHT): action = "right"
        if pr.is_key_down(pr.KEY_LEFT): action = "left"
        if pr.is_key_down(pr.KEY_SPACE): action = "release"

        obs, reward, done = env.step(action, PHYSICS_DT)

        if action == "release":
            print(f"RELEASE STATE: {obs}")

        sim_time += PHYSICS_DT

        if done:
            real_time = time.perf_counter() - start_time
            # Print episode summary
            env.print_episode_summary(sim_time, real_time)
            print(f"COLLISION STATE: {obs}")
            print(f"COLLISION REWARD: {reward}")
            print(f"SUCCESS: {env.success}")
            # Reset environment
            obs = env.reset(phi=PHI)

        # Rendering
        env.render()
        time.sleep(0.05)

    pr.close_window()