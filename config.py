# Game Specific Configuration File Created by User

# -- methods -- #

# returns true if state is terminal
def is_terminal_state(state):
    pass

# returns next actions from state -- shouldn't be called on terminal state
def get_actions(state):
    pass

# returns normalized policy distribution over legal actions
def get_policy(policy_network,state):
    pass

# returns rewards for given state, should include terminal rewards
def evaluate_state(value_network,state):
    pass

# evaluate preformance of policy network and value network
def evaluate_preformance(policy_network, value_network):
    pass
