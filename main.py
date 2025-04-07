import human_control
import train

MODE = "HUMAN"

if __name__ == "__main__":
    if MODE == "HUMAN":
        human_control.main()
    elif MODE == "TRAIN":
        train.main()