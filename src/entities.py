import pyray as pr
from typing import NoReturn
import math

class Target:
    def __init__(self, position: pr.Vector3, radius: float = 5.0):
        self.pos: pr.Vector3 = position
        self.radius = radius

class Drone:
    """Represents a drone agent in the simulation environment.
    
    Attributes:
        pos (pr.Vector3): Current position in 3D space (x,y,z)
        speed (float): Movement speed in units per second
        size (pr.Vector3): Dimensions of drone collision box (width, height, depth)
    """
    
    def __init__(self, position: pr.Vector3) -> None:
        """Initialize drone with starting position.
        
        Args:
            position: Initial 3D position (x,y,z) of the drone
        """
        self.pos: pr.Vector3 = position
        self.speed: float = 10.0  # Movement speed in units/second
        self.size: pr.Vector3 = pr.Vector3(10.0, 5.0, 10.0)  # Width, height, depth

    def update(self, action: str, dt: float) -> None:
        """Update drone position based on movement action.
        
        Args:
            action: Movement command (forward/backward/left/right)
            dt: Time delta since last update in seconds
            
        Raises:
            ValueError: If invalid action is provided
        """
        movement = self.speed * dt
        
        if action == "forward":
            self.pos.x += movement
        elif action == "backward":
            self.pos.x -= movement
        elif action == "left":
            self.pos.z -= movement  # Note: Z-axis is depth in 3D space
        elif action == "right":
            self.pos.z += movement
        else:
            self._handle_invalid_action(action)

    def relative_position(self, other: pr.Vector3) -> pr.Vector3:
        """Calculate position of another object relative to the drone.
        
        Args:
            other: Absolute position of another object
            
        Returns:
            Relative position vector from drone to other object
        """
        return pr.vector3_subtract(other, self.pos)

    def _handle_invalid_action(self, action: str) -> NoReturn:
        """Handle invalid movement actions.
        
        Args:
            action: The invalid action string
            
        Raises:
            ValueError: Always raises with description of invalid action
        """
        raise ValueError(f"Invalid drone action: '{action}'. "
                       f"Valid actions are: forward, backward, left, right")
    
class Grenade:
    def __init__(self, position: pr.Vector3):
        self.pos = position 
        self.vel = pr.Vector3(0.0, 0.0, 0.0)
        self.size = pr.Vector3(5.0, 5.0, 5.0)
        self.is_released = False
        self.mass = 0.4
        self.drag_coef = 0.47
        self.air_density = 1.225
        self.cross_sectional_area = math.pi * (0.032 / 2)**2

    def update(self, dt: float, gravity: pr.Vector3, wind: pr.Vector3, drone_pos: pr.Vector3):
        if not self.is_released:
            self.pos = pr.Vector3(
                drone_pos.x,
                drone_pos.y - 2,
                drone_pos.z
            )
            self.vel = pr.Vector3(0, 0, 0)
        else:
            gravitational_force = pr.vector3_scale(gravity, self.mass)  # F = m*g
            relative_velocity = pr.vector3_subtract(self.vel, wind)  # Velocity relative to wind
            
            # Calculate aerodynamic drag force
            drag_force = pr.Vector3(0, 0, 0)
            if pr.vector3_length(relative_velocity) > 0:
                drag_direction = pr.vector3_normalize(relative_velocity)
                drag_direction = pr.vector3_scale(drag_direction, -1.0)
                speed = pr.vector3_length(relative_velocity)
                drag_magnitude = 0.5 * self.air_density * self.drag_coef * self.cross_sectional_area * speed * speed
                drag_force = pr.vector3_scale(drag_direction, drag_magnitude)
            
            # Combine forces and calculate acceleration
            net_force = pr.vector3_add(gravitational_force, drag_force)
            acceleration = pr.vector3_scale(net_force, 1/self.mass)
            
            # Update velocity and position using Euler integration
            self.vel = pr.vector3_add(self.vel, pr.vector3_scale(acceleration, dt))
            self.pos = pr.vector3_add(self.pos, pr.vector3_scale(self.vel, dt))

    def release(self):
        if not self.is_released:
            self.is_released = True
