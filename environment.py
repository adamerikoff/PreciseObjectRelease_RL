import random
import pyray as pr
import entities

class Environment:
    def __init__(self, scene_size: pr.Vector3):
        self.scene_size = scene_size
        half_size = pr.Vector3(scene_size.x/2, scene_size.y/2, scene_size.z/2)
        
        # Generate random positions within bounds
        target_position = self._get_random_position(half_size, min_distance=150)
        # Drone starts near target but in the air
        drone_position = pr.Vector3(
            target_position.x + random.uniform(-5, 5),
            random.uniform(half_size.y, scene_size.y),  # Ensure drone is in upper half
            target_position.z + random.uniform(-5, 5)
        )
        # Grenade starts just below drone
        grenade_position = pr.Vector3(drone_position.x, drone_position.y - 1, drone_position.z)

        self.target = entities.Target(target_position, pr.Vector3(10, 10, 10))
        self.drone = entities.Drone(drone_position, pr.Vector3(10, 4, 10))
        self.grenade = entities.Grenade(grenade_position, pr.Vector3(5, 5, 5))

        # Camera setup - position it to see the whole scene
        self.camera = pr.Camera3D(
            pr.Vector3(scene_size.x, scene_size.y, scene_size.z),   # position
            pr.Vector3(half_size.x, half_size.y + 50, half_size.z), # target
            pr.Vector3(0, 1, 0),                                    # up
            45.0,                                                   # fovy
            pr.CAMERA_PERSPECTIVE                                   # projection
        )

    def update(self, delta_time: float):
        self.drone.update(delta_time)
        self.grenade.update(delta_time, 0.0, pr.Vector3(0, 0, 0), self.drone)
        # if not self.grenade.is_released:
        #     self.grenade.follow_drone(self.drone.position)
        
        # if pr.is_key_pressed(pr.KEY_SPACE):
        #     self.grenade.release()
        
        # if self.grenade.is_released:
        #     self.grenade.update(delta_time, gravity=pr.Vector3(0, -9.8, 0), wind=pr.Vector3(0, 0, 0))

    def draw(self, delta_time: float):
        """Render the entire 3D scene with debug information."""
        # 3D Rendering
        pr.begin_mode_3d(self.camera)
        
        # Draw the ground grid (scaled to scene size)
        grid_subdivisions = int(self.scene_size.x / 10)
        grid_spacing = 10.0
        pr.draw_grid(grid_subdivisions, grid_spacing)
        
        # Draw all game entities
        self._draw_entities()
        
        pr.end_mode_3d()

        # 2D Overlay (debug info)
        self._draw_debug_info(delta_time)

    def _draw_entities(self):
        """Draw all 3D objects in the scene."""
        # Draw target (green cube)
        pr.draw_cube(self.target.position, 
                    self.target.size.x, 
                    self.target.size.y, 
                    self.target.size.z, 
                    self.target.color)

        # Draw drone (blue cube)
        pr.draw_cube(self.drone.position, 
                    self.drone.size.x, 
                    self.drone.size.y, 
                    self.drone.size.z, 
                    self.drone.color)

        # Draw grenade
        pr.draw_cube(self.grenade.position, 
                    self.grenade.size.x, 
                    self.grenade.size.y, 
                    self.grenade.size.z, 
                    self.grenade.color)

    def _draw_debug_info(self, delta_time: float):
        """Draw debug information as 2D overlay."""
        # Drone position display
        drone_pos_text = (
            f"Drone Position: "
            f"{self.drone.position.x:.1f}, "
            f"{self.drone.position.y:.1f}, "
            f"{self.drone.position.z:.1f}"
        )
        pr.draw_text(drone_pos_text, 10, 10, 20, pr.BLACK)
        # Grenade position display
        grenade_pos_text = (
            f"Grenade Position: "
            f"{self.grenade.position.x:.1f}, "
            f"{self.grenade.position.y:.1f}, "
            f"{self.grenade.position.z:.1f}"
        )
        pr.draw_text(grenade_pos_text, 10, 30, 20, pr.BLACK)
        # Target position display
        target_pos_text = (
            f"Target Position: "
            f"{self.target.position.x:.1f}, "
            f"{self.target.position.y:.1f}, "
            f"{self.target.position.z:.1f}"
        )
        pr.draw_text(target_pos_text, 10, 50, 20, pr.BLACK)
        
        # Grenade state display
        grenade_state = "Released" if self.grenade.is_released else "Attached"
        pr.draw_text(f"Grenade State: {grenade_state}", 10, 70, 20, pr.BLACK)
        
        velocity_text = (
                f"Velocity: "
                f"{self.drone.movement_vector.x:.1f}, "
                f"{self.drone.movement_vector.y:.1f}, "
                f"{self.drone.movement_vector.z:.1f}"
            )
        pr.draw_text(velocity_text, 10, 90, 20, pr.BLACK)

        delta_time_text = (
                f"Delta Time: "
                f"{delta_time:.5f}, "
            )
        pr.draw_text(delta_time_text, 10, 110, 20, pr.BLACK)

    def _get_random_position(self, half_size: pr.Vector3, min_distance: float = 0) -> pr.Vector3:
        """Helper to get random position within bounds, ensuring min_distance from edges."""
        return pr.Vector3(
            random.uniform(-half_size.x + min_distance, half_size.x - min_distance),
            0,  # On ground
            random.uniform(-half_size.z + min_distance, half_size.z - min_distance)
        )

