import control_human
import control_dqn

MODE = "DQN"

if __name__ == "__main__":
    if MODE == "HUMAN":
        control_human.main()
    elif MODE == "DQN":
        control_dqn.main()
