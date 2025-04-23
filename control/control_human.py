"""
Main execution script for human-controlled drone grenade delivery simulation.
Provides keyboard controls and real-time visualization of the environment.
"""

import time
from typing import Optional, Tuple, List

import pyray as pr

from src.environment import Environment
from src.renderer import Renderer
from config.config import SCENE_X, SCENE_Z, SCENE_Y, SCREEN_HEIGHT, SCREEN_WIDTH

def main() -> None:
    """Main execution function for the human-controlled simulation."""
    # Window configuration
    WINDOW_TITLE: str = "Human Control"

    # Physics timestep (fixed for stability)
    PHYSICS_DT: float = 0.05

    # Environment initialization
    env: Environment = Environment((SCENE_X, SCENE_Y, SCENE_Z))
    obs: List[float] = env.reset()
    
    # Simulation state
    done: bool = False

    # Rendering setup
    renderer: Renderer = Renderer(SCREEN_WIDTH, SCREEN_HEIGHT, WINDOW_TITLE, env)
    renderer.window_init()

    # Main game loop
    while not done and not pr.window_should_close():
        # Process keyboard input
        action: Optional[str] = None
        if pr.is_key_down(pr.KEY_UP): 
            action = "forward"
        if pr.is_key_down(pr.KEY_DOWN): 
            action = "backward"
        if pr.is_key_down(pr.KEY_RIGHT): 
            action = "right"
        if pr.is_key_down(pr.KEY_LEFT): 
            action = "left"
        if pr.is_key_down(pr.KEY_SPACE): 
            action = "release"

        # Environment step
        obs, reward, done = env.step(action, PHYSICS_DT)

        # Episode handling
        if done:
            env.print_episode_summary(reward)
            obs = env.reset()

        # Rendering and frame pacing
        renderer.render()
        time.sleep(0.02)

    # Cleanup
    renderer.close_window()


if __name__ == "__main__":
    main()