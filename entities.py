import pyray as pr
import numpy as np
import math

class Drone:
    def __init__(self, position: pr.Vector3):
        # Initialize drone with position, movement speed, and physical dimensions
        self.pos = position  # 3D position in world space (x,y,z)
        self.speed = 5.0  # Movement speed in units per second
        self.size = pr.Vector3(5.0, 2.0, 5.0)  # Dimensions (width, height, depth)
        
    def update(self, action: str, dt: float):
        """Update position based on action (positive x is forward)
        Args:
            action: Movement direction command
            dt: Time delta since last update (for frame-rate independent movement)
        """
        # Move drone based on input command and elapsed time
        if action == "forward":
            self.pos.x += self.speed * dt  # Move forward along x-axis
        elif action == "backward":
            self.pos.x -= self.speed * dt  # Move backward along x-axis
        elif action == "left":
            self.pos.z -= self.speed * dt  # Move left along z-axis
        elif action == "right":
            self.pos.z += self.speed * dt  # Move right along z-axis
        else:
            print(f"Error: Invalid drone action: {action}")  # Handle invalid inputs

    def relative_position(self, other: pr.Vector3) -> pr.Vector3:
        # Calculate position relative to another point in space
        return pr.vector3_subtract(other, self.pos)

class Grenade:
    def __init__(self, position: pr.Vector3):
        # Initialize grenade physics properties
        self.pos = position  # 3D position in world space
        self.vel = pr.Vector3(0.0, 0.0, 0.0)  # Velocity vector
        self.size = pr.Vector3(2.0, 2.0, 2.0)  # Physical dimensions
        self.is_released = False  # Release state flag
        self.mass = 0.4  # Mass in kilograms
        self.drag_coef = 0.47  # Aerodynamic drag coefficient
        self.air_density = 1.225  # Air density at sea level (kg/m³)
        self.cross_sectional_area = math.pi * (0.032 / 2)**2  # Frontal area for drag calculation

    def update(self, dt: float, gravity: pr.Vector3, wind: pr.Vector3, drone_pos: pr.Vector3):
        """Update grenade physics
        Args:
            dt: Time delta for physics simulation
            gravity: Gravity force vector
            wind: Wind velocity vector
            drone_pos: Current drone position (for unreleased state)
        """
        if not self.is_released:
            # If not released, follow drone position (slightly below it)
            self.pos = pr.Vector3(
                drone_pos.x,  # Match drone x position
                drone_pos.y - 2,  # Hang slightly below drone
                drone_pos.z  # Match drone z position
            )
            self.vel = pr.Vector3(0, 0, 0)  # Zero velocity when carried
        else:
            # Physics simulation when released
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
        """Trigger grenade release from drone"""
        if not self.is_released:
            self.is_released = True  # Set release flag

class Target:
    def __init__(self, position: pr.Vector3):
        # Initialize target with position and size
        self.pos = position  # 3D position in world space
        self.size = pr.Vector3(4.0, 4.0, 4.0)  # Physical dimensions