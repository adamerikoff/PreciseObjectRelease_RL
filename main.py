import control_human
import control_dqn_jump
import control_dqn
import control_test

MODE = "TEST_MODEL"

if __name__ == "__main__":
    if MODE == "HUMAN":
        control_human.main()
    elif MODE == "DQN_JUMP":
        control_dqn_jump.main()
    elif MODE == "DQN_VANILLA":
        control_dqn.main()
    elif MODE == "TEST_MODEL":
        control_test.main()
    
