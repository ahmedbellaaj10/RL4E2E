import random
"""
The first two actions are from https://github.com/cycraig/MP-DQN
"""

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


def get_random_actions(number_actions , top_k):
    return random.choice(len(number_actions) , top_k)

def get_actions(q_val , top_k):
    return (-q_val).argsort()[:top_k]