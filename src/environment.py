from typing import Tuple, List, Optional, Dict, Any
import math

import pyray as pr
import numpy as np

from src.objects import Drone, Ball, Target

class Environment:
    def __init__(self, scene_size: np.ndarray[np.float64], ) -> None:
        self.scene_size: np.ndarray[np.float64] = scene_size
        self.half_scene_size: np.ndarray[np.float64] = scene_size / 2

        self.state_size: int = 8
        self.action_size: int = 5

    def reset(self, height: Optional[float] = None) -> np.ndarray[np.float64]:
        self.gravity: np.ndarray[np.float64] = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        self.wind: np.ndarray[np.float64] = np.array([
            np.random.uniform(-10.0, 10.0),  # x: random between -20 and 20
            0.0,                              # y: fixed at 0
            np.random.uniform(-10.0, 10.0)   # z: random between -20 and 20
        ])

        target_pos = np.array([
            np.random.uniform(-self.half_scene_size[0] + 100, self.half_scene_size[0] - 100),
            0.0,                             
            np.random.uniform(-self.half_scene_size[2] + 100, self.half_scene_size[2] - 100)
        ], dtype=np.float64)
        drone_pos = np.array([
            target_pos[0] + np.random.uniform(-10.0, 10.0),
            height if height is not None else np.random.uniform(50.0, self.scene_size[1]),                             
            target_pos[2] + np.random.uniform(-10.0, 10.0)  # Fixed index from [3] to [2]
        ], dtype=np.float64)
        ball_pos = np.array([
            drone_pos[0],
            drone_pos[1] - 2.0,
            drone_pos[2]
        ], dtype=np.float64)

        self.target = Target(target_pos)
        self.drone = Drone(drone_pos)
        self.ball = Ball(ball_pos)

        self.episode_time: float = 0.0
        self.free_fall_time: float = 0.0
        self.episode_reward: float = 0.0
        self.episode_steps: int = 0
        self.success: bool = False
        
        self.action_count: Dict = {
            "forward": 0,    
            "backward": 0,   
            "left": 0,       
            "right": 0,      
            "release": 0,
        }

        return self.get_obs()

    def step(self, action: Optional[int], dt: float) -> Tuple[np.ndarray[np.float64], float, float]:
        if action is not None: 
            if action == 0: self.action_count["forward"] += 1
            if action == 1: self.action_count["backward"] += 1
            if action == 2: self.action_count["left"] += 1
            if action == 3: self.action_count["right"] += 1
            if action == 4: self.action_count["release"] += 1
        if action is not None:
            if action == 4:
                self.ball.release()
            else:
                self.drone.update(action, dt)

        self.ball.update(self.gravity, self.wind, self.drone.position, dt)
        
        done: bool = self.check_done()
        reward: float = self.calculate_reward()
        
        self.episode_time += dt
        self.episode_steps += 1

        if self.ball.is_released:
            self.free_fall_time += dt

        self.episode_reward += reward

        return (
            self.get_obs(),      # Flattened observations (1D array)
            reward, float(done)  # Reward + done flag
        )
    
    def check_done(self) -> bool:
        return self.ball.position[1] <= 0.0
    
    def calculate_reward(self) -> float:
        wind_magnitude: float = np.linalg.norm(self.wind)
        current_distance: float = np.linalg.norm(self.target.position - self.ball.position)
        if self.check_done():
            if current_distance <= self.target.radius:
                if current_distance <= 1.0: return 100
                self.success = True
                distance_ratio = 1 - current_distance/self.target.radius
                height_ratio = 1 - self.drone.position[1]/self.scene_size[1]
                return (1 + wind_magnitude + distance_ratio + height_ratio)
            return -(1 + current_distance/self.scene_size[0])
        return -0.01
    
    def calculate_ball_target_angle(self) -> float:
        ball_vec: np.ndarray[np.float64] = self.ball.position - self.drone.position
        target_vec: np.ndarray[np.float64] = self.target.position - self.drone.position

        dot_product: float = np.dot(ball_vec, target_vec)
        norm_grenade: float = np.linalg.norm(ball_vec)
        norm_target: float = np.linalg.norm(target_vec)

        if norm_grenade == 0 or norm_target == 0:
            return 0.0
        
        cos_theta: float = np.clip(dot_product / (norm_grenade * norm_target), -1.0, 1.0)
        return math.acos(cos_theta)
    
    def calculate_ball_target_distance(self) -> float:
        ball_vec: np.ndarray[np.float64] = self.ball.position - self.drone.position
        target_vec: np.ndarray[np.float64] = self.target.position - self.drone.position
        return np.linalg.norm(target_vec - ball_vec)
    
    def get_obs(self) -> np.ndarray[np.float64]:
        target_vec: np.ndarray[np.float64] = self.target.position - self.drone.position
        relative_distance: float = self.calculate_ball_target_distance()
        angle_rad: float = self.calculate_ball_target_angle()

        return np.array(
            [
                self.drone.position[1],
                self.wind[0],                # Wind x
                self.wind[2],                # Wind z
                target_vec[0],               # Target rel x
                target_vec[2],               # Target rel z
                relative_distance,           # Distance to target
                angle_rad,
                float(self.ball.is_released) # Release status
            ],
            dtype=np.float64  # Explicitly enforce float64
        )
    
    def print_episode_summary(self, collision_reward: float) -> None:
        episode_time: str = f"{self.episode_time:.2f}"
        free_fall_time: str = f"{self.free_fall_time:.4f}"
        total_reward: str = f"{self.episode_reward:7.2f}"
        steps: str = f"{self.episode_steps:7.2f}"

        separator: str = "═" * 60
        output: List[str] = [
            separator,
            f"TOTAL FALL TIME:  {free_fall_time}s",
            f"TOTAL EPISODE TIME:  {episode_time}s",
            f"TOTAL REWARD:     {total_reward}",
            f"TOTAL STEPS:      {steps}s",
            f"COLLISION REWARD: {collision_reward}",
            f"SUCCESS:          {self.success}",
            f"AC: f={self.action_count['forward']} b={self.action_count['backward']} l={self.action_count['left']} r={self.action_count['right']} r={self.action_count['release']}",
            separator
        ]
        print("\n".join(output))

        
        

        
