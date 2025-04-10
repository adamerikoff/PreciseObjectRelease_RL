import control_human
import control_dqn_jump
import control_dqn

MODE = "DQN_JUMP"

if __name__ == "__main__":
    if MODE == "HUMAN":
        control_human.main()
    elif MODE == "DQN_JUMP":
        control_dqn_jump.main()
    elif MODE == "DQN_VANILLA":
        control_dqn.main()
