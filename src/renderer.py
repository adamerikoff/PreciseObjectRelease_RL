"""
3D rendering module for drone grenade delivery simulation.
Handles visualization of the environment and debug information.
"""

import pyray as pr
from typing import List, Optional

from src.environment import Environment

class Renderer:
    """Handles 3D visualization and debug information display for the simulation."""
    
    def __init__(self, 
                 screen_width: int, 
                 screen_height: int, 
                 title: str, 
                 env: Environment, 
                 top: bool = False, 
                 debug: bool = True) -> None:
        """Initialize the renderer with display settings and environment reference.
        
        Args:
            screen_width: Window width in pixels
            screen_height: Window height in pixels
            title: Window title
            env: Environment instance to render
            top: Whether to use top-down camera view
            debug: Whether to show debug information
        """
        self.screen_width: int = screen_width
        self.screen_height: int = screen_height
        self.title: str = title
        self.env: Environment = env
        self.debug: bool = debug
        self.camera: pr.Camera3D = self.setup_camera_top_view() if top else self.setup_camera_side_view()

    def window_init(self) -> None:
        """Initialize the rendering window."""
        pr.init_window(self.screen_width, self.screen_height, self.title)

    def close_window(self) -> None:
        """Close the rendering window."""
        pr.close_window()

    def render(self) -> None:
        """Render one frame of the simulation."""
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        # 3D scene rendering
        pr.begin_mode_3d(self.camera)
        self._render_3d_scene()
        pr.end_mode_3d()
        
        # Debug information overlay
        if self.debug:
            self.draw_debug_info()
            
        pr.end_drawing()

    def _render_3d_scene(self) -> None:
        """Render all 3D elements of the simulation."""
        # Draw grid (10m squares)
        grid_spacing: int = int(self.env.scene_size.x / 10)
        pr.draw_grid(grid_spacing, 10)
        
        # Draw entities
        pr.draw_cube(self.env.drone.pos, 
                    self.env.drone.size.x, 
                    self.env.drone.size.y, 
                    self.env.drone.size.z, 
                    pr.BLUE)
        
        pr.draw_cube(self.env.grenade.pos, 
                    self.env.grenade.size.x, 
                    self.env.grenade.size.y, 
                    self.env.grenade.size.z, 
                    pr.RED)
        
        pr.draw_cylinder(
            self.env.target.pos,
            self.env.target.radius,
            self.env.target.radius,
            1.0,  # Tiny height (almost flat)
            16,    # Number of segments (smoothness)
            (0, 255, 0, 128)
        )

    def setup_camera_side_view(self) -> pr.Camera3D:
        """Configure side-view camera.
        
        Returns:
            Camera3D instance positioned for side view
        """
        return pr.Camera3D(
            pr.Vector3(
                self.env.scene_size.x, 
                self.env.scene_size.y, 
                self.env.scene_size.z
            ),
            pr.Vector3(
                self.env.half_size.x, 
                self.env.half_size.y + 100, 
                self.env.half_size.z
            ),
            pr.Vector3(0, 1, 0),
            45.0,
            pr.CAMERA_PERSPECTIVE
        )

    def setup_camera_top_view(self) -> pr.Camera3D:
        """Configure top-down camera.
        
        Returns:
            Camera3D instance positioned for top view
        """
        return pr.Camera3D(
            pr.Vector3(
                self.env.drone.pos.x, 
                self.env.drone.pos.y + 50, 
                self.env.drone.pos.z + 5
            ),
            pr.Vector3(
                self.env.drone.pos.x, 
                self.env.drone.pos.y, 
                self.env.drone.pos.z
            ),
            pr.Vector3(0, 0, 1),  # Z-axis is up in top-down view
            45.0,
            pr.CAMERA_PERSPECTIVE
        )

    def draw_debug_info(self) -> None:
        """Render debug information overlay."""
        # Calculate debug values
        distance: float = self.env.calculate_grenade_target_distance()
        target_rel_pos: pr.Vector3 = self.env.drone.relative_position(self.env.target.pos)
        grenade_rel_pos: pr.Vector3 = self.env.drone.relative_position(self.env.grenade.pos)
        angle: float = self.env.calculate_grenage_target_angle()

        # Prepare debug text lines
        debug_text: List[str] = [
            f"Sim Timer: {self.env.episode_time:.2f}",
            f"Free Fall Timer: {self.env.free_fall_time:.2f}",
            f"Wind: ({self.env.wind.x:.1f}, {self.env.wind.y:.1f}, {self.env.wind.z:.1f})",
            f"Gravity: ({self.env.gravity.x:.1f}, {self.env.gravity.y:.1f}, {self.env.gravity.z:.1f})",
            f"Drone Pos: ({self.env.drone.pos.x:.1f}, {self.env.drone.pos.y:.1f}, {self.env.drone.pos.z:.1f})",
            f"Target Pos: ({self.env.target.pos.x:.1f}, {self.env.target.pos.y:.1f}, {self.env.target.pos.z:.1f})",
            f"Target Rel Pos: ({target_rel_pos.x:.1f}, {target_rel_pos.y:.1f}, {target_rel_pos.z:.1f})",
            f"Grenade Pos: ({self.env.grenade.pos.x:.1f}, {self.env.grenade.pos.y:.1f}, {self.env.grenade.pos.z:.1f})",
            f"Grenade Rel Pos: ({grenade_rel_pos.x:.1f}, {grenade_rel_pos.y:.1f}, {grenade_rel_pos.z:.1f})",
            f"Grenade Vel: ({self.env.grenade.vel.x:.1f}, {self.env.grenade.vel.y:.1f}, {self.env.grenade.vel.z:.1f})",
            f"Grenade/Target Distance: {distance:.2f}",
            f"Grenade/Target Angle: {angle:.2f} rad",
            f"Steps: {self.env.episode_steps}",
            f"Total Reward: {self.env.episode_reward:.2f}",
        ]

        # Render debug text
        for i, text in enumerate(debug_text):
            y_pos: int = 40 + i * 25
            pr.draw_text(text, 15, y_pos, 20, pr.BLACK)