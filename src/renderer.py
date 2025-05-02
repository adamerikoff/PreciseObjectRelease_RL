import time
from typing import List, Optional

import numpy as np
import pyray as pr

from src.environment import Environment

class Renderer:
    def __init__(self, 
        screen_width: int, 
        screen_height: int, 
        title: str, 
        env: Environment, 
        top: bool = False, 
        debug: bool = True,
        rendering_sleep: float = 0.5
        ) -> None:
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.title: str = title
        self.env: Environment = env
        self.debug: bool = debug
        self.top = top
        self.rendering_sleep: float = rendering_sleep

    def window_init(self) -> None:
        pr.init_window(self.screen_width, self.screen_height, self.title)

    def close_window(self) -> None:
        pr.close_window()

    def render(self) -> None:
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        pr.begin_mode_3d(self.setup_camera_top_view() if self.top else self.setup_camera_side_view())
        self._render_3d_scene()
        pr.end_mode_3d()
        
        # Debug information overlay
        if self.debug:
            self.draw_debug_info()
            
        pr.end_drawing()
        time.sleep(self.rendering_sleep)

    def _render_3d_scene(self) -> None:
        grid_spacing: int = int(self.env.scene_size[0] / 10)
        pr.draw_grid(grid_spacing, 10)
        
        pr.draw_cube(self.env.drone.position.tolist(), 
                    0 if self.top else self.env.drone.size[0], 
                    0 if self.top else self.env.drone.size[1], 
                    0 if self.top else self.env.drone.size[2], 
                    pr.BLUE)
        
        pr.draw_cube(self.env.ball.position.tolist(), 
                    self.env.ball.size[0], 
                    self.env.ball.size[1], 
                    self.env.ball.size[2], 
                    pr.RED)
        
        pr.draw_cylinder(
            self.env.target.position.tolist(),
            self.env.target.radius,
            self.env.target.radius,
            1.0,  
            16,
            (0, 255, 0, 128)
        )

    def setup_camera_side_view(self) -> pr.Camera3D:
        return pr.Camera3D(
            pr.Vector3(
                self.env.scene_size[0], 
                self.env.scene_size[1] + 50, 
                self.env.scene_size[2]
            ),
            pr.Vector3(
                self.env.half_scene_size[0], 
                self.env.half_scene_size[1], 
                self.env.half_scene_size[2]
            ),
            pr.Vector3(0, 1, 0),
            45.0,
            pr.CAMERA_PERSPECTIVE
        )
    
    def setup_camera_top_view(self) -> pr.Camera3D:
        return pr.Camera3D(
            pr.Vector3(
                self.env.drone.position[0], 
                self.env.drone.position[1] + 100, 
                self.env.drone.position[2]
            ),
            pr.Vector3(
                self.env.drone.position[0], 
                self.env.drone.position[1], 
                self.env.drone.position[2]
            ),
            pr.Vector3(0, 0, 1),  # Z-axis is up in top-down view
            45.0,
            pr.CAMERA_PERSPECTIVE
        )
    
    def draw_debug_info(self) -> None:
        """Render debug information overlay."""
        # Calculate debug values
        distance: float = self.env.calculate_ball_target_distance()
        target_rel_vec: np.ndarray = self.env.target.position - self.env.drone.position
        grenade_rel_vec: np.ndarray = self.env.ball.position - self.env.drone.position
        angle: float = self.env.calculate_ball_target_angle()

        # Prepare debug text lines
        debug_text: List[str] = [
            f"Sim Timer: {self.env.episode_time:.2f}",
            f"Free Fall Timer: {self.env.free_fall_time:.2f}",
            f"Wind: ({self.env.wind[0]:.1f}, 0.0, {self.env.wind[2]:.1f})",
            f"Gravity: ({self.env.gravity[0]:.1f}, {self.env.gravity[1]:.1f}, {self.env.gravity[2]:.1f})",
            f"Drone Pos: ({self.env.drone.position[0]:.1f}, {self.env.drone.position[1]:.1f}, {self.env.drone.position[2]:.1f})",
            f"Target Pos: ({self.env.target.position[0]:.1f}, {self.env.target.position[1]:.1f}, {self.env.target.position[2]:.1f})",
            f"Target Rel Pos: ({target_rel_vec[0]:.1f}, {target_rel_vec[1]:.1f}, {target_rel_vec[2]:.1f})",
            f"Grenade Pos: ({self.env.ball.position[0]:.1f}, {self.env.ball.position[1]:.1f}, {self.env.ball.position[2]:.1f})",
            f"Grenade Rel Pos: ({grenade_rel_vec[0]:.1f}, {grenade_rel_vec[1]:.1f}, {grenade_rel_vec[2]:.1f})",
            f"Grenade Vel: ({self.env.ball.velocity[0]:.1f}, {self.env.ball.velocity[1]:.1f}, {self.env.ball.velocity[2]:.1f})",
            f"Grenade/Target Distance: {distance:.2f}",
            f"Grenade/Target Angle: {angle:.2f} rad",
            f"Steps: {self.env.episode_steps}",
            f"Total Reward: {self.env.episode_reward:.2f}",
        ]

        # Render debug text
        for i, text in enumerate(debug_text):
            y_pos: int = 40 + i * 25
            pr.draw_text(text, 15, y_pos, 20, pr.BLACK)

