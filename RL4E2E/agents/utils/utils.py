import random
import numpy as np

"""
The first two actions are from https://github.com/cycraig/MP-DQN
"""

def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


def get_actions(Q_a, top_k):
    Q_a_array = np.array(Q_a)
    top_indexes = (-Q_a_array).argsort()[:top_k]
    top_qa = [Q_a_array[i] for i in top_indexes]
    top_percentage = np.round(top_qa/ sum(top_qa),2)
    actions = []
    j = 0
    for i in range(len(Q_a)):
        if i in top_indexes:
            actions.extend([i, top_percentage[j]])
            j = j+1
        else :
            actions.extend([i, 0.])
    return actions

def get_random_actions(n_actions, top_k):
    Q_a = [random.uniform(0,10) for _ in range(n_actions)]
    actions = get_actions(Q_a, top_k)
    return actions