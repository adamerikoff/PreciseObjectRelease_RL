import random
import pyray as pr
import math
import environment

# main.py
def main():
    pr.init_window(1200, 800, "Drone Grenade Environment")
    pr.set_target_fps(60)

    env = environment.Environment((500, 400, 500))
    obs = env.reset()

    while not pr.window_should_close():
        dt = 0.1
        action = random.randint(0, 4)
        obs, reward, done, info = env.step(action, dt)
        env.render()
        # print(obs)
        if done:
            print(f"EPISODE FINISHED: {env.episode_time:.2f}")
            print(f"THEORETICAL TIME {(2 * env.drone.pos.y / -env.gravity.y)**0.5:.2f}")
            obs = env.reset()


    pr.close_window()

if __name__ == "__main__":
    main()