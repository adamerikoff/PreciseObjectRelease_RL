import time
from typing import Optional

import numpy as np
import pyray as pr

from src.environment import Environment
from src.renderer import Renderer
from CONFIG import SCENE_X, SCENE_Z, SCENE_Y, SCREEN_HEIGHT, SCREEN_WIDTH, PHYSICS_DT, RENDERING_SLEEP

def main() -> None:
    WINDOW_TITLE: str = "Human Control"

    env: Environment = Environment(np.array([SCENE_X, SCENE_Y, SCENE_Z]))
    obs: np.ndarray[np.float64] = env.reset()
    
    done: bool = False

    renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE, env, True, True, RENDERING_SLEEP)
    renderer.window_init()

    while not done and not pr.window_should_close():
        # Process keyboard input
        action: Optional[str] = None
        if pr.is_key_down(pr.KEY_UP): 
            action = 0
        if pr.is_key_down(pr.KEY_DOWN): 
            action = 1
        if pr.is_key_down(pr.KEY_RIGHT): 
            action = 2
        if pr.is_key_down(pr.KEY_LEFT): 
            action = 3
        if pr.is_key_down(pr.KEY_SPACE): 
            action = 4

        state, reward, done = env.step(action, PHYSICS_DT)

        done = bool(done)

        if done:
            env.print_episode_summary(reward)
            obs = env.reset()

        renderer.render()

    renderer.close_window()


if __name__ == "__main__":
    main()