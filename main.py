import random
import pyray as pr
import environment

# main.py
def main():
    pr.init_window(1200, 800, "Drone Grenade Environment")
    pr.set_target_fps(60)

    env = environment.Environment((400, 500, 400))
    obs = env.reset()

    while not pr.window_should_close():
        dt = pr.get_frame_time()
        action = random.randint(0, 3)
        obs, reward, done, info = env.step(action, dt)
        env.render()

        if done:
            obs = env.reset()

    pr.close_window()

if __name__ == "__main__":
    main()