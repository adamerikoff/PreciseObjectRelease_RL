import human_control
import train_control

MODE = "TRAIN"

if __name__ == "__main__":
    if MODE == "HUMAN":
        human_control.main()
    elif MODE == "TRAIN":
        train_control.main()