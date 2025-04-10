import random
import math
from typing import Tuple, Dict

import numpy as np
import pyray as pr
import entities

class Environment:
    def __init__(self, scene_size: Tuple[float, float, float]):
        self.state_size = 8
        self.action_size = 5
        self.action_space = [
            "forward",
            "backward",
            "left",
            "right",
            "release",
        ]

        self.scene_size = pr.Vector3(*scene_size)
        self.half_size = pr.vector3_scale(scene_size, 0.5)
        
        self.gravity = None
        self.wind = None
        
        self.target = None
        self.drone = None
        self.grenade = None
        
        self.episode_time = None
        self.free_fall_time = None
        
        self.total_reward = None

        self.steps = None
        self.prev_distance = None

        self.success = False

    def reset(self, phi=False):
        self.gravity = pr.Vector3(0.0, -9.81, 0.0)

        self.wind = pr.Vector3(
            random.uniform(-10.0, 10.0) * (1 - phi),
            0.0,
            random.uniform(-10.0, 10.0) * (1 - phi)
        )

        self.target = entities.Target(
            pr.Vector3(
                random.uniform(-self.half_size.x + 100, self.half_size.x - 100),
                0.0,  # On ground level
                random.uniform(-self.half_size.z + 100, self.half_size.z - 100)
            )
        )

        self.drone = entities.Drone(
            pr.Vector3(
                self.target.pos.x + random.uniform(-10.0, 10.0) * (1 - phi),
                random.randrange(50, self.scene_size.y),
                self.target.pos.z + random.uniform(-10.0, 10.0) * (1 - phi)
            )
        )

        self.grenade = entities.Grenade(
            pr.Vector3(
                self.drone.pos.x,
                self.drone.pos.y - 2.0,
                self.drone.pos.z
            )
        )

        self.episode_time = 0.0
        self.free_fall_time = 0.0
        self.total_reward = 0.0
        self.steps = 0

        self.prev_distance = None
        self.success = False

        return self._get_obs()

    def step(self, action: str, dt: float) -> Tuple[list, float, bool]:
        if action:
            if action == "release":
                self.grenade.release()
            else:
                self.drone.update(action, dt)

        self.grenade.update(dt, self.gravity, self.wind, self.drone.pos)
        
        done = self._check_done()
        reward = self._calculate_reward()
        
        self.episode_time += dt
        self.steps += 1

        if self.grenade.is_released:
            self.free_fall_time += dt

        self.total_reward += reward

        return self._get_obs(), reward, done

    def simulate_free_fall(self, dt: float) -> Tuple[list, float, bool, int]:
        
        self.grenade.release() 

        reward_accumulator = 0

        while not self._check_done():
            self.grenade.update(dt, self.gravity, self.wind, self.drone.pos)
            reward = self._calculate_reward()
            
            self.total_reward += reward
            reward_accumulator += reward
            
            self.free_fall_time += dt
            self.episode_time += dt
            self.steps += 1
            
        return self._get_obs(), reward_accumulator, self._check_done(), self.steps

    def _get_obs(self) -> list:
        target_vec = self.drone.relative_position(self.target.pos)
        
        relative_distance = self._calculate_grenade_target_distance()
        angle_rad = self._calculate_grenage_target_angle()

        return [
            self.drone.pos.y, 
            self.wind.x, self.wind.z,
            target_vec.x, target_vec.z,
            relative_distance,
            angle_rad,
            1 if self.grenade.is_released else 0
        ]


    def _calculate_grenade_target_distance(self) -> float:
        grenade_vec = self.drone.relative_position(self.grenade.pos)
        target_vec = self.drone.relative_position(self.target.pos)

        return pr.vector3_length(pr.vector3_subtract(target_vec, grenade_vec))
    
    def _calculate_grenage_target_angle(self) -> float:
        grenade_vec = self.drone.relative_position(pr.Vector3(
            self.drone.pos.x,
            self.drone.pos.y - 2,
            self.drone.pos.z,
        ))
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
        SUCCESS_RADIUS = 5.0
        current_distance = pr.vector3_distance(self.target.pos, self.grenade.pos)
        wind_magnitude = np.sqrt(self.wind.x**2 + self.wind.z**2)
        
        if self._check_done():
            if current_distance <= SUCCESS_RADIUS:
                self.success = True
                
                return 1 + wind_magnitude + (1 - current_distance/SUCCESS_RADIUS) + (1 - self.drone.pos.y/self.scene_size.y)
            return -(1 + current_distance/self.scene_size.x)

        return -0.01

    def _check_done(self) -> bool:
        if self.grenade.pos.y <= 0.0:
            return True
        return False

    def render(self):
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)
        
        pr.begin_mode_3d(self._setup_camera())
        
        pr.draw_grid(int(self.scene_size.x/10), 10)

        pr.draw_cube(self.drone.pos, self.drone.size.x, self.drone.size.y, self.drone.size.z, pr.BLUE)        
        pr.draw_cube(self.grenade.pos, self.grenade.size.x, self.grenade.size.y, self.grenade.size.z, pr.RED)
        pr.draw_cube(self.target.pos, self.target.size.x, self.target.size.y, self.target.size.z, pr.GREEN)
        
        pr.end_mode_3d()
        self._draw_debug_info()
        pr.end_drawing()

    def _setup_camera(self) -> pr.Camera3D:
        camera = pr.Camera3D(
            pr.Vector3(self.scene_size.x, self.scene_size.y, self.scene_size.z),
            pr.Vector3(self.half_size.x, self.half_size.y + 100, self.half_size.z),
            pr.Vector3(0, 1, 0),
            45.0,
            pr.CAMERA_PERSPECTIVE
        )
        return camera
    
    def _draw_debug_info(self):
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
            f"Steps: ({self.steps})",
            f"Total Reward: ({self.total_reward})",
        ]

        for i, text in enumerate(debug_text):
            y_pos = 40 + i * 25
            pr.draw_text(text, 15, y_pos, 20, pr.BLACK)

    def print_episode_summary(self, real_time_elapsed: float, total_real_time: float):
        speed_ratio = self.episode_time / real_time_elapsed
        sim_time = f"{self.episode_time:>10.2f}"
        free_fall_time = f"{self.free_fall_time:>7.4f}"
        real_time = f"{real_time_elapsed:>7.4f}"
        speed = f"{speed_ratio:>5.2f}"
        total_time = f"{total_real_time:>7.2f}"
        total_reward = f"{self.total_reward:>7.2f}"
        steps = f"{self.steps:>7.2f}"

        separator = "═" * 60
        output = [
            separator,
            f"⏱️ FREEFALL TIME: {free_fall_time}s",
            f"⏱️ SIM TIME: {sim_time}s",
            f"🕒 REAL TIME: {real_time}s (Wall Clock)  |  {speed}x Speed",
            f"Σ TOTAL REAL TIME: {total_time}s",
            f"Σ TOTAL REWARD: {total_reward}",
            f"Σ TOTAL STEPS: {steps}s",
            separator
        ]

        print("\n".join(output))