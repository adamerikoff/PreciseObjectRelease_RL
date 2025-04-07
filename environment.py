import random
import math
from typing import Tuple, Dict

import numpy as np
import pyray as pr
import entities

class Environment:
    def __init__(self, scene_size: Tuple[float, float, float]):
        self.scene_size = pr.Vector3(*scene_size)
        self.half_size = pr.vector3_scale(scene_size, 0.5)
        self.wind = None
        self.gravity = None
        self.drone = None
        self.grenade = None
        self.target = None
        self.episode_time = None
        self.theoretical_time_required = None

    def reset(self):
        self.gravity = pr.Vector3(0.0, -9.81, 0.0)
        self.wind = pr.Vector3(
            random.uniform(-10.0, 10.0),
            0.0,
            random.uniform(-10.0, 10.0)
        )
        self.target = entities.Target(
            pr.Vector3(
                random.uniform(-self.half_size.x + 100, self.half_size.x - 100),
                0.0,
                random.uniform(-self.half_size.z + 100, self.half_size.z - 100)
            )
        )
        self.drone = entities.Drone(
            pr.Vector3(
                self.target.pos.x + random.uniform(-3.0, 3.0), 
                random.randrange(100, self.scene_size.y), 
                self.target.pos.z + random.uniform(-3.0, 3.0)
            )
        )
        self.grenade = entities.Grenade(
            pr.Vector3(
                self.drone.pos.x,
                self.drone.pos.y - 1.0,
                self.drone.pos.z
            )
        )

        self.episode_time = 0.0
        self.theoretical_time_required = (2*self.drone.pos.y/-self.gravity.y)**0.5

        return self._get_obs()

    def step(self, action: int, dt: float) -> Tuple[Dict, float, bool, Dict]:
        if action:
            if action == 0:
                self.drone.update("forward", dt)
            elif action == 1:
                self.drone.update("backward", dt)
            elif action == 2:
                self.drone.update("left", dt)
            elif action == 3:
                self.drone.update("right", dt)
            elif action == 4:
                self.grenade.release()

        self.grenade.update(dt, self.gravity, self.wind, self.drone.pos)
        
        done = self._check_done()
        reward = self._calculate_reward()
        
        self.episode_time += dt

        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> Dict:
        grenade_vec = np.array([
            self.drone.relative_position(self.grenade.pos).x,
            self.drone.relative_position(self.grenade.pos).y,
            self.drone.relative_position(self.grenade.pos).z
        ])

        target_vec = np.array([
            self.drone.relative_position(self.target.pos).x,
            self.drone.relative_position(self.target.pos).y,
            self.drone.relative_position(self.target.pos).z
        ])
        
        dot_product = np.dot(grenade_vec, target_vec)
        norm_grenade = np.linalg.norm(grenade_vec)
        norm_target = np.linalg.norm(target_vec)
        
        if norm_grenade == 0 or norm_target == 0:
            angle_rad = 0.0
        else:
            cos_theta = dot_product / (norm_grenade * norm_target)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = math.acos(cos_theta)
        
        relative_distance = np.linalg.norm(target_vec - grenade_vec)
        
        return {
            "drone_relative_pos": (0.0, 0.0, 0.0),
            "grenade_relative_pos": tuple(grenade_vec),
            "target_relative_pos": tuple(target_vec),
            "relative_distance": relative_distance,
            "angle_grenade_to_target": angle_rad,
            "grenade_released": self.grenade.is_released
        }

    def _calculate_reward(self):
        return 1.0

    def _check_done(self):
        if self.grenade.pos.y <= 0.0:
            return True
        return False

    def render(self):
        """Centralized rendering of all entities"""
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        pr.begin_mode_3d(self._setup_camera())
        
        pr.draw_grid(int(self.scene_size.x/10), 10)

        pr.draw_cube(self.drone.pos, self.drone.size.x, self.drone.size.y, self.drone.size.z, pr.BLUE)        
        pr.draw_cube(self.grenade.pos, self.grenade.size.x, self.grenade.size.y, self.grenade.size.z, pr.RED)
        pr.draw_cube(self.target.pos, self.target.size.x, self.target.size.y, self.target.size.z, pr.GREEN)
        
        pr.end_mode_3d()
        self._print_debug_info()
        pr.end_drawing()

    def _setup_camera(self) -> pr.Camera3D:
        camera = pr.Camera3D(
            pr.Vector3(self.scene_size.x, self.scene_size.y, self.scene_size.z),   # position
            pr.Vector3(self.half_size.x, self.half_size.y + 100, self.half_size.z), # target
            pr.Vector3(0, 1, 0),                                    # up
            45.0,                                                   # fovy
            pr.CAMERA_PERSPECTIVE                                   # projection
        )
        return camera
    
    def _print_debug_info(self):
        debug_text = [
            f"Sim Timer: {self.episode_time}",
            f"Wind: ({self.wind.x:.1f}, {self.wind.y:.1f}, {self.wind.z:.1f})",
            f"Gravity: ({self.gravity.x:.1f}, {self.gravity.y:.1f}, {self.gravity.z:.1f})",
            f"Drone Pos: ({self.drone.pos.x:.1f}, {self.drone.pos.y:.1f}, {self.drone.pos.z:.1f})",
            f"Target Pos: ({self.target.pos.x:.1f}, {self.target.pos.y:.1f}, {self.target.pos.z:.1f})",
            f"Grenade Pos: ({self.grenade.pos.x:.1f}, {self.grenade.pos.y:.1f}, {self.grenade.pos.z:.1f})",
            f"Grenade Vel: ({self.grenade.vel.x:.1f}, {self.grenade.vel.y:.1f}, {self.grenade.vel.z:.1f})",
            f"Theoretical Fall Time: {self.theoretical_time_required:.2f}s"
        ]

        for i, text in enumerate(debug_text):
            y_pos = 40 + i * 25  # 25px per line
            pr.draw_text(text, 15, y_pos, 20, pr.BLACK)
