from training import train_base

train_base.train("DQN_JUMP", "dqn_jump", render=False, jump=True, top=False)

train_base.train("DQN", "dqn", render=False, jump=False, top=True)