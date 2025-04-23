"""
Environment module for drone grenade delivery simulation.
Handles the physics, state management, and reward calculation for the training environment.
"""

import random
import math
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pyray as pr

from src.entities import Grenade, Target, Drone

class Environment:
    """
    A 3D environment for simulating drone-based grenade delivery to targets.
    Handles physics simulation, state observation, and reward calculation.
    """
    
    def __init__(self, scene_size: Tuple[float, float, float]) -> None:
        """Initialize the environment with given dimensions.
        
        Args:
            scene_size: Tuple of (width, height, depth) for environment bounds
        """
        # Environment dimensions
        self.state_size: int = 8  # Dimension of observation space
        self.action_size: int = 5  # Dimension of action space
        self.action_space: List[str] = [
            "forward",    # Move drone forward
            "backward",   # Move drone backward
            "left",       # Move drone left
            "right",      # Move drone right
            "release",   # Release grenade
        ]

        # Spatial properties
        self.scene_size: pr.Vector3 = pr.Vector3(*scene_size)
        self.half_size: pr.Vector3 = pr.vector3_scale(self.scene_size, 0.5)
        
        # Physics properties
        self.gravity: Optional[pr.Vector3] = None
        self.wind: Optional[pr.Vector3] = None
        
        # Entities
        self.target: Optional[Target] = None
        self.drone: Optional[Drone] = None
        self.grenade: Optional[Grenade] = None
        
        # Episode tracking
        self.episode_time: Optional[float] = None
        self.free_fall_time: Optional[float] = None
        self.episode_reward: Optional[float] = None
        self.episode_steps: Optional[int] = None
        self.success: bool = False

    def reset(self, height: Optional[float] = None) -> List[float]:
        """Reset the environment to initial state.
        
        Args:
            height: Optional starting height for the drone. If None, random height is used.
            
        Returns:
            Initial observation vector
        """
        # Physics setup
        self.gravity = pr.Vector3(0.0, -9.81, 0.0)
        self.wind = pr.Vector3(
            random.uniform(-10.0, 10.0) * (1),
            0.0,
            random.uniform(-10.0, 10.0) * (1)
        )

        # Target placement (on ground)
        self.target = Target(
            pr.Vector3(
                random.uniform(-self.half_size.x + 100, self.half_size.x - 100),
                0.0,
                random.uniform(-self.half_size.z + 100, self.half_size.z - 100)
            )
        )

        # Drone placement (near target)
        self.drone = Drone(
            pr.Vector3(
                self.target.pos.x + random.uniform(-10.0, 10.0),
                height if height else random.randrange(100, self.scene_size.y),
                self.target.pos.z + random.uniform(-10.0, 10.0)
            )
        )

        # Grenade placement (just below drone)
        self.grenade = Grenade(
            pr.Vector3(
                self.drone.pos.x,
                self.drone.pos.y - 1.0,
                self.drone.pos.z
            )
        )

        # Reset episode tracking
        self.episode_time: float = 0.0
        self.free_fall_time: float = 0.0
        self.episode_reward: float = 0.0
        self.episode_steps: int = 0
        self.success: bool = False

        return self.get_obs()

    def step(self, action: Optional[str], dt: float) -> Tuple[List[float], float, bool]:
        """Execute one timestep of the environment.
        
        Args:
            action: The action to take (from action_space)
            dt: Time delta for physics simulation
            
        Returns:
            Tuple of (observation, reward, done)
        """
        if action:
            if action == "release":
                self.grenade.release()
            else:
                self.drone.update(action, dt)

        self.grenade.update(dt, self.gravity, self.wind, self.drone.pos)
        
        done: bool = self.check_done()
        reward: float = self.calculate_reward()
        
        self.episode_time += dt
        self.episode_steps += 1

        if self.grenade.is_released:
            self.free_fall_time += dt

        self.episode_reward += reward

        return self.get_obs(), reward, done

    def get_obs(self) -> List[float]:
        """Get current observation vector.
        
        Returns:
            List containing:
            - Drone y position
            - Wind x and z components
            - Target relative x and z position
            - Grenade-target distance
            - Grenade-target angle (radians)
            - Grenade release status (1 if released)
        """
        target_vec: pr.Vector3 = self.drone.relative_position(self.target.pos)
        relative_distance: float = self.calculate_grenade_target_distance()
        angle_rad: float = self.calculate_grenage_target_angle()

        return [
            self.drone.pos.y, 
            self.wind.x, self.wind.z,
            target_vec.x, target_vec.z,
            relative_distance,
            angle_rad,
            1.0 if self.grenade.is_released else 0.0
        ]

    def calculate_grenade_target_distance(self) -> float:
        """Calculate horizontal distance between grenade and target.
        
        Returns:
            Distance in meters
        """
        grenade_vec: pr.Vector3 = self.drone.relative_position(self.grenade.pos)
        target_vec: pr.Vector3 = self.drone.relative_position(self.target.pos)
        return pr.vector3_length(pr.vector3_subtract(target_vec, grenade_vec))
    
    def calculate_grenage_target_angle(self) -> float:
        """Calculate angle between grenade drop vector and target vector.
        
        Returns:
            Angle in radians
        """
        grenade_vec: pr.Vector3 = self.drone.relative_position(pr.Vector3(
            self.drone.pos.x,
            self.drone.pos.y - 2,
            self.drone.pos.z,
        ))
        target_vec: pr.Vector3 = self.drone.relative_position(self.target.pos)

        dot_product: float = pr.vector3_dot_product(grenade_vec, target_vec)
        norm_grenade: float = pr.vector3_length(grenade_vec)
        norm_target: float = pr.vector3_length(target_vec)
        
        if norm_grenade == 0 or norm_target == 0:
            return 0.0
            
        cos_theta: float = np.clip(dot_product / (norm_grenade * norm_target), -1.0, 1.0)
        return math.acos(cos_theta)

    def calculate_reward(self) -> float:
        """Calculate reward for current state.
        
        Returns:
            Reward value (higher is better)
        """
        SUCCESS_RADIUS: float = 5.0
        current_distance: float = pr.vector3_distance(self.target.pos, self.grenade.pos)
        wind_magnitude: float = np.sqrt(self.wind.x**2 + self.wind.z**2)
        
        if self.check_done():
            if current_distance <= SUCCESS_RADIUS:
                self.success = True
                return (1 + wind_magnitude + 
                       (1 - current_distance/SUCCESS_RADIUS) + 
                       (1 - self.drone.pos.y/self.scene_size.y))
            return -(1 + current_distance/self.scene_size.x)

        return -0.01  # Small penalty for each timestep

    def check_done(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode should end (grenade hit ground)
        """
        return self.grenade.pos.y <= 0.0
    
    def print_episode_summary(self, collision_reward: float) -> None:
        """Print summary statistics for completed episode.
        
        Args:
            collision_reward: Additional reward component to display
        """
        episode_time: str = f"{self.episode_time:.2f}"
        free_fall_time: str = f"{self.free_fall_time:.4f}"
        total_reward: str = f"{self.episode_reward:7.2f}"
        steps: str = f"{self.episode_steps:7.2f}"

        separator: str = "═" * 60
        output: List[str] = [
            separator,
            f"TOTAL FALL TIME:  {free_fall_time}s",
            f"TOTAL REAL TIME:  {episode_time}s",
            f"TOTAL REWARD:     {total_reward}",
            f"TOTAL STEPS:      {steps}s",
            f"COLLISION REWARD: {collision_reward}",
            f"SUCCESS:          {self.success}",
            separator
        ]

        print("\n".join(output))