import pyray as pr
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
        return pr.Vector3(
            other.x - self.pos.x,
            other.y - self.pos.y,
            other.z - self.pos.z,
        )

class Grenade:
    def __init__(self, position: pr.Vector3):
        self.pos = position
        self.vel = pr.Vector3(0.0, 0.0, 0.0)
        self.size = pr.Vector3(2.0, 2.0, 2.0)
        self.is_released = False
        
    def update(self, dt: float, gravity: pr.Vector3, wind: pr.Vector3, drone_pos: pr.Vector3):
        if not self.is_released:
            self.pos = pr.Vector3(
                drone_pos.x,
                drone_pos.y - 2,
                drone_pos.z
            )
        else:
            self.vel.x = self.vel.x + (gravity.x + wind.x) * dt
            self.vel.y = self.vel.y + (gravity.y + wind.y) * dt
            self.vel.z = self.vel.z + (gravity.z + wind.z) * dt
            self.pos.x = self.pos.x + self.vel.x * dt
            self.pos.y = self.pos.y + self.vel.y * dt
            self.pos.z = self.pos.z + self.vel.z * dt

    def release(self):
        if not self.is_released:
            self.is_released = True

class Target:
    def __init__(self, position: pr.Vector3):
        self.pos = position
        self.size = pr.Vector3(4.0, 4.0, 4.0)
