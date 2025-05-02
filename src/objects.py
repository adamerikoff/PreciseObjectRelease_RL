import numpy as np
import math
from typing import Final

class Target:
    def __init__(
            self, 
            position: np.ndarray[np.float64] = np.array([0.0, 0.0, 0.0], dtype=np.float64), 
            radius: float = 5.0
        ) -> None:
        self.position: np.ndarray[np.float64] = position.astype(np.float64)
        self.radius: float = radius


class Drone:
    def __init__(
            self,
            position: np.ndarray[np.float64] = np.array([0.0, 0.0, 0.0], dtype=np.float64),
            size: np.ndarray[np.float64] = np.array([10.0, 5.0, 10.0], dtype=np.float64),
            speed: float = 10.0
        ) -> None:
        self.position: np.ndarray[np.float64] = position.astype(np.float64)
        self.size: np.ndarray[np.float64] = size.astype(np.float64)
        self.speed: float = speed
    
    def update(self, action: int, dt: float) -> None:
        movement: np.ndarray[np.float64] = np.zeros(3, dtype=np.float64)
        
        if action == 0:
            movement[0] = self.speed * dt
        elif action == 1:
            movement[0] = -self.speed * dt
        elif action == 2:
            movement[2] = -self.speed * dt
        elif action == 3:
            movement[2] = self.speed * dt
            
        self.position += movement


class Ball:
    DEFAULT_MASS: Final[float] = 0.4
    DEFAULT_DRAG_COEF: Final[float] = 0.47
    DEFAULT_AIR_DENSITY: Final[float] = 1.225
    DEFAULT_RADIUS: Final[float] = 0.032
    
    def __init__(
            self,
            position: np.ndarray[np.float64] = np.array([0.0, 0.0, 0.0], dtype=np.float64),
            size: np.ndarray[np.float64] = np.array([5.0, 5.0, 5.0], dtype=np.float64),
            velocity: np.ndarray[np.float64] = np.array([0.0, 0.0, 0.0], dtype=np.float64),
            is_released: bool = False,
            mass: float = DEFAULT_MASS,
            drag_coef: float = DEFAULT_DRAG_COEF,
            air_density: float = DEFAULT_AIR_DENSITY,
            radius: float = DEFAULT_RADIUS
        ) -> None:
        self.position: np.ndarray[np.float64] = position.astype(np.float64)
        self.size: np.ndarray[np.float64] = size.astype(np.float64)
        self.velocity: np.ndarray[np.float64] = velocity.astype(np.float64)
        self.is_released: bool = is_released
        self.mass: float = mass
        self.drag_coef: float = drag_coef
        self.air_density: float = air_density
        self.radius: float = radius
        self.cross_sectional_area: float = math.pi * radius**2

    def update(
            self, 
            gravity: np.ndarray[np.float64], 
            wind: np.ndarray[np.float64], 
            drone_position: np.ndarray[np.float64], 
            dt: float
        ) -> None:
        if not self.is_released:
            self.position = drone_position + np.array([0.0, -1.0, 0.0], dtype=np.float64)
            self.velocity = np.zeros(3, dtype=np.float64)
        else:
            gravitational_force: np.ndarray[np.float64] = gravity * self.mass
            relative_velocity: np.ndarray[np.float64] = self.velocity - wind
            speed: float = np.linalg.norm(relative_velocity)

            drag_force: np.ndarray[np.float64] = np.zeros(3, dtype=np.float64)
            if speed > 0.0:
                drag_direction: np.ndarray[np.float64] = relative_velocity / speed
                drag_magnitude: float = (
                    0.5 * self.air_density * (speed ** 2) * 
                    self.drag_coef * self.cross_sectional_area
                )
                drag_force = -drag_direction * drag_magnitude
            
            net_force: np.ndarray[np.float64] = gravitational_force + drag_force
            acceleration: np.ndarray[np.float64] = net_force / self.mass

            self.velocity += acceleration * dt
            self.position += self.velocity * dt

    def release(self):
        if not self.is_released: self.is_released = True
