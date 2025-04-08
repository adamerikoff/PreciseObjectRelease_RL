import random
import math
from typing import Tuple, Dict

import numpy as np
import pyray as pr
import entities

class Environment:
    def __init__(self, scene_size: Tuple[float, float, float], enable_wind=True):
        # Initialize environment with 3D scene dimensions (width, height, depth)
        self.scene_size = pr.Vector3(*scene_size)  # Full dimensions of the environment
        self.half_size = pr.vector3_scale(scene_size, 0.5)  # Half dimensions for calculations
        self.wind = None  # Will store wind velocity vector
        self.gravity = None  # Will store gravity vector
        self.drone = None  # Drone entity placeholder
        self.grenade = None  # Grenade entity placeholder
        self.target = None  # Target entity placeholder
        self.episode_time = None  # Timer for current simulation episode
        self.theoretical_time_required = None  # Calculated free-fall time
        self.free_fall_time = None
        self.total_reward = None
        self.enable_wind = enable_wind
        self.state_size = 9
        self.action_size = 5
        self.action_space = [
            "forward",
            "backward",
            "left",
            "right",
            "release",
        ]

    def reset(self):
        """Reset environment to initial state for new episode"""
        # Set constant downward gravity (Earth standard)
        self.gravity = pr.Vector3(0.0, -9.81, 0.0)
        
        if self.enable_wind == False:
            self.wind = pr.Vector3(
                0.0,
                0.0,
                0.0
            )
        else:
            # Initialize random wind with horizontal components only
            self.wind = pr.Vector3(
                random.uniform(-10.0, 10.0),  # Random x-component
                0.0,  # No vertical wind
                random.uniform(-10.0, 10.0)  # Random z-component
            )

        # Place target on ground within bounds (keeping 100 unit margin from edges)
        self.target = entities.Target(
            pr.Vector3(
                random.uniform(-self.half_size.x + 100, self.half_size.x - 100),
                0.0,  # On ground level
                random.uniform(-self.half_size.z + 100, self.half_size.z - 100)
            )
        )

        # Initialize drone near target position but at random height
        self.drone = entities.Drone(
            pr.Vector3(
                self.target.pos.x + random.uniform(-100.0, 100.0),  # X near target
                random.randrange(100, self.scene_size.y),  # Random height
                self.target.pos.z + random.uniform(-100.0, 100.0)  # Z near target
            )
        )

        # Initialize grenade just below drone
        self.grenade = entities.Grenade(
            pr.Vector3(
                self.drone.pos.x,
                self.drone.pos.y - 1.0,  # 1 unit below drone
                self.drone.pos.z
            )
        )

        # Reset episode timer
        self.episode_time = 0.0
        self.free_fall_time = 0.0

        self.total_reward = 0.0

        # Calculate theoretical free-fall time from drone height (t = √(2h/g))
        self.theoretical_time_required = (2*self.drone.pos.y/-self.gravity.y)**0.5

        return self._get_obs()

    def step(self, action: str, dt: float) -> Tuple[list, float, bool]:
        """Execute one timestep of the environment
        Args:
            action: str command for drone control
            dt: Timestep duration in seconds
        Returns:
            Tuple containing (observation, reward, done flag, info dict)
        """
        # Process drone movement actions if provided
        if action:
            if action == "release":
                self.grenade.release()  # Release grenade
            else:
                self.drone.update(action, dt)

        # Update grenade physics (affected by gravity and wind)
        self.grenade.update(dt, self.gravity, self.wind, self.drone.pos)
        
        # Check termination conditions
        done = self._check_done()
        # Calculate reward for this step
        reward = self._calculate_reward()
        
        # Increment episode timer
        self.episode_time += dt

        if self.grenade.is_released:
            self.free_fall_time += dt

        self.total_reward += reward

        return self._get_obs(), reward, done

    def _get_obs(self) -> list:
        """Generate observation dictionary containing environment state"""
        grenade_vec = self.drone.relative_position(self.grenade.pos)
        target_vec = self.drone.relative_position(self.target.pos)
        
        relative_distance = self._calculate_grenade_target_distance()
        angle_rad = self._calculate_grenage_target_angle()

        return [
            grenade_vec.x, grenade_vec.y, grenade_vec.z,
            target_vec.x, target_vec.y, target_vec.z,
            relative_distance,
            angle_rad,
            1 if self.grenade.is_released else 0
        ]


    def _calculate_grenade_target_distance(self) -> float:
        grenade_vec = self.drone.relative_position(self.grenade.pos)
        target_vec = self.drone.relative_position(self.target.pos)

        return pr.vector3_length(pr.vector3_subtract(target_vec, grenade_vec))
    
    def _calculate_grenage_target_angle(self) -> float:
        grenade_vec = self.drone.relative_position(self.grenade.pos)
        target_vec = self.drone.relative_position(self.target.pos)

        dot_product = pr.vector3_dot_product(grenade_vec, target_vec)

        norm_grenade = pr.vector3_length(grenade_vec)
        norm_target = pr.vector3_length(target_vec)
        
        if norm_grenade == 0 or norm_target == 0:
            angle_rad = 0.0
        else:
            cos_theta = dot_product / (norm_grenade * norm_target)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = math.acos(cos_theta)
        
        return angle_rad

    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        reward = -0.1
        if self._check_done():
            distance = pr.vector3_distance(self.target.pos, self.grenade.pos)
            if distance <= 5.0:
                reward += 10
            else:
                reward += -distance/10
        return reward

    def _check_done(self) -> bool:
        """Check termination conditions for current episode"""
        if self.grenade.pos.y <= 0.0:  # Episode ends when grenade hits ground
            return True
        return False

    def render(self):
        """Centralized rendering of all entities"""
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)  # Clear screen with white
        
        # Start 3D rendering mode with configured camera
        pr.begin_mode_3d(self._setup_camera())
        
        # Draw grid for visual reference (spaced every 10 units)
        pr.draw_grid(int(self.scene_size.x/10), 10)

        # Render all entities as colored cubes
        pr.draw_cube(self.drone.pos, self.drone.size.x, self.drone.size.y, self.drone.size.z, pr.BLUE)        
        pr.draw_cube(self.grenade.pos, self.grenade.size.x, self.grenade.size.y, self.grenade.size.z, pr.RED)
        pr.draw_cube(self.target.pos, self.target.size.x, self.target.size.y, self.target.size.z, pr.GREEN)
        
        pr.end_mode_3d()  # End 3D rendering
        self._draw_debug_info()  # Display debug text
        pr.end_drawing()  # Finish drawing frame

    def _setup_camera(self) -> pr.Camera3D:
        """Configure 3D camera perspective"""
        camera = pr.Camera3D(
            pr.Vector3(self.scene_size.x, self.scene_size.y, self.scene_size.z),   # Camera position (top-right-back corner)
            pr.Vector3(self.half_size.x, self.half_size.y + 100, self.half_size.z), # Look-at target (center of scene, slightly elevated)
            pr.Vector3(0, 1, 0),                                    # Up vector (world Y-axis)
            45.0,                                                   # Field of view (degrees)
            pr.CAMERA_PERSPECTIVE                                   # Perspective projection
        )
        return camera
    
    def _draw_debug_info(self):
        """Display debug information as on-screen text"""
        distance = self._calculate_grenade_target_distance()
        target_rel_pos = self.drone.relative_position(self.target.pos)
        grenade_rel_pos = self.drone.relative_position(self.grenade.pos)

        debug_text = [
            f"Sim Timer: {self.episode_time}",
            f"Free Fall Timer: {self.free_fall_time}",
            f"Wind: ({self.wind.x:.1f}, {self.wind.y:.1f}, {self.wind.z:.1f})",
            f"Gravity: ({self.gravity.x:.1f}, {self.gravity.y:.1f}, {self.gravity.z:.1f})",
            f"Drone Pos: ({self.drone.pos.x:.1f}, {self.drone.pos.y:.1f}, {self.drone.pos.z:.1f})",
            f"Target Pos: ({self.target.pos.x:.1f}, {self.target.pos.y:.1f}, {self.target.pos.z:.1f})",
            f"Target Rel Pos: ({target_rel_pos.x:.1f}, {target_rel_pos.y:.1f}, {target_rel_pos.z:.1f})",
            f"Grenade Pos: ({self.grenade.pos.x:.1f}, {self.grenade.pos.y:.1f}, {self.grenade.pos.z:.1f})",
            f"Grenade Rel Pos: ({grenade_rel_pos.x:.1f}, {grenade_rel_pos.y:.1f}, {grenade_rel_pos.z:.1f})",
            f"Grenade Vel: ({self.grenade.vel.x:.1f}, {self.grenade.vel.y:.1f}, {self.grenade.vel.z:.1f})",
            f"Grenade/Target Distance: ({distance})",
            f"Grenade/Target Angle: ({self._calculate_grenage_target_angle()})",
            f"Theoretical Fall Time: {self.theoretical_time_required:.2f}s"
        ]

        # Render each debug line with 25px vertical spacing
        for i, text in enumerate(debug_text):
            y_pos = 40 + i * 25  # 25px per line
            pr.draw_text(text, 15, y_pos, 20, pr.BLACK)

    def print_episode_summary(self, real_time_elapsed: float, total_real_time: float):
        speed_ratio = self.episode_time / real_time_elapsed
        sim_time = f"{self.episode_time:>10.2f}"
        theory_time = f"{self.theoretical_time_required:>7.2f}"
        free_fall_time = f"{self.free_fall_time:>7.4f}"
        real_time = f"{real_time_elapsed:>7.4f}"
        speed = f"{speed_ratio:>5.2f}"
        total_time = f"{total_real_time:>7.2f}"
        total_reward = f"{self.total_reward:>7.2f}"

        separator = "═" * 60
        output = [
            separator,
            f"⏱️ FREEFALL TIME: {free_fall_time}s  |  THEORETICAL: {theory_time}s",
            f"⏱️ SIM TIME: {sim_time}s",
            f"🕒 REAL TIME: {real_time}s (Wall Clock)  |  {speed}x Speed",
            f"Σ TOTAL REAL TIME: {total_time}s",
            f"Σ TOTAL REWARD: {total_reward}s",
            separator
        ]

        print("\n".join(output))