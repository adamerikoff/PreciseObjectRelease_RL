import pyray as pr

class Object:
    def __init__(self, position: pr.Vector3, size: pr.Vector3, color: pr.Color):
        self.position = position
        self.size = size
        self.color = color

    def update(self):
        """Update logic (e.g., movement, collisions)."""
        pass

class Drone(Object):
    def __init__(self, position: pr.Vector3, size: pr.Vector3):
        super().__init__(position, size, pr.BLUE)
        self.movement_vector = pr.Vector3(0.0, 0.0, 0.0)

    def update(self, delta_time: float):
        # Correct way to update position with movement vector
        new_x = self.position.x + self.movement_vector.x * delta_time
        new_y = self.position.y + self.movement_vector.y * delta_time
        new_z = self.position.z + self.movement_vector.z * delta_time
        self.position = pr.Vector3(new_x, new_y, new_z)

class Grenade(Object):
    def __init__(self, position: pr.Vector3, size: pr.Vector3):
        super().__init__(position, size, pr.RED)
        self.is_released = False
        self.drag_coefficient = 0.47
        self.mass = 0.4

    def check_collision(self):
        if self.position.y <= 0:
            return True
        return False

    def update(self, delta_time: float, gravity: float, wind: pr.Vector3, drone: Drone):
        if self.is_released:
            self.position = self.position + delta_time * (gravity + wind)
        else:
            self.follow_drone(drone.position)
    
    def follow_drone(self, drone_position: pr.Vector3):
        self.position.x = drone_position.x
        self.position.y = drone_position.y - 1
        self.position.z = drone_position.z
    
    def release(self):
        if not self.is_released:
            self.is_released = True

class Target(Object):
    def __init__(self, position: pr.Vector3, size: pr.Vector3):
        super().__init__(position, size, pr.GREEN)

