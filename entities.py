import pyray as pr
import numpy as np
import math

class Drone:
    def __init__(self, position: pr.Vector3):
        self.pos = position
        self.speed = 5.0
        self.size = pr.Vector3(5.0, 2.0, 5.0)
        
    def update(self, action: str, dt: float):
        """Update position based on action (positive x is forward)"""
        if action == "forward":
            self.pos.x += self.speed * dt
        elif action == "backward":
            self.pos.x -= self.speed * dt
        elif action == "left":
            self.pos.z -= self.speed * dt
        elif action == "right":
            self.pos.z += self.speed * dt
        else:
            print(f"Error: Invalid drone action: {action}")

    def relative_position(self) -> pr.Vector3:
        return pr.Vector3(0.0, 0.0, 0.0)

    def relative_position(self, other: pr.Vector3) -> pr.Vector3:
        return pr.vector3_subtract(other, self.pos)

class Grenade:
    def __init__(self, position: pr.Vector3):
        self.pos = position
        self.vel = pr.Vector3(0.0, 0.0, 0.0)
        self.size = pr.Vector3(2.0, 2.0, 2.0)
        self.is_released = False
        self.mass = 0.4  # kg
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
            gravitational_force = pr.vector3_scale(gravity, self.mass)
            relative_velocity = pr.vector3_subtract(self.vel, wind)
            drag_force = pr.Vector3(0, 0, 0)
            if pr.vector3_length(relative_velocity) > 0:
                drag_force = pr.vector3_normalize(drag_force)
                drag_force = pr.vector3_scale(drag_force, self.air_density * -0.5 * self.drag_coef * self.cross_sectional_area * pr.vector3_length(relative_velocity))
            net_force = pr.vector3_add(gravitational_force, drag_force)
            acceleration = pr.vector3_scale(net_force, 1/self.mass)
            self.vel = pr.vector3_add(self.vel, pr.vector3_scale(acceleration, dt))
            self.pos = pr.vector3_add(self.pos, pr.vector3_scale(self.vel, dt))

    def release(self):
        if not self.is_released:
            self.is_released = True

class Target:
    def __init__(self, position: pr.Vector3):
        self.pos = position
        self.size = pr.Vector3(4.0, 4.0, 4.0)
