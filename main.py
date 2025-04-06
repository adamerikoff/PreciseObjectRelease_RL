import pyray as pr
import environment

def main():
    pr.init_window(800, 600, "3D Scene Example")
    pr.set_target_fps(60)
    
    scene = environment.Environment(pr.Vector3(500, 400, 500))
    
    while not pr.window_should_close():
        delta_time = pr.get_frame_time()
        
        scene.update(delta_time)
        
        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)
        scene.draw(delta_time)
        pr.end_drawing()

    
    pr.close_window()

if __name__ == "__main__":
    main()