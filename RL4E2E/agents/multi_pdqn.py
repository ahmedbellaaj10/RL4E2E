import torch
import torch.nn.functional as F
import numpy as np
from RL4E2E.environemnt.multiwoz_simulator import MultiwozSimulator
from RL4E2E.agents.utils.utils import get_random_actions, get_actions
import argparse
from RL4E2E.agents.pdqn import QActor, ParamActor, PDQNAgent


class MultiPDQN(PDQNAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 top_k,
                 actor_class=QActor,
                 actor_kwargs={'hidden_layers': (128,),
                               'action_input_layer': 0},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={'hidden_layers': (128,),
                                     'squashing_function': False,
                                     'output_layer_init_std': 0.0001},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.9,
                 tau_actor=0.01,
                 tau_actor_param=0.001,
                 replay_memory_size=10000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.0001,
                 initial_memory_threshold= 50,
                 use_ornstein_noise=False,
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=True,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device='cpu',
                 seed=1):
        super().__init__(observation_space,
                         action_space,
                         actor_class=actor_class,
                         actor_kwargs=actor_kwargs,
                         actor_param_class=actor_param_class,
                         actor_param_kwargs=actor_param_kwargs,
                         epsilon_initial=epsilon_initial,
                         epsilon_final=epsilon_final,
                         epsilon_steps=epsilon_steps,
                         batch_size=batch_size,
                         gamma=gamma,
                         tau_actor=tau_actor,
                         tau_actor_param=tau_actor_param,
                         replay_memory_size=replay_memory_size,
                         learning_rate_actor=learning_rate_actor,
                         learning_rate_actor_param=learning_rate_actor_param,
                         initial_memory_threshold=initial_memory_threshold,
                         use_ornstein_noise=use_ornstein_noise,
                         loss_func=loss_func,
                         clip_grad=clip_grad,
                         inverting_gradients=inverting_gradients,
                         zero_index_gradients=zero_index_gradients,
                         indexed=indexed,
                         weighted=weighted,
                         average=average,
                         random_weighted=random_weighted,
                         device=device,
                         seed=seed)

        self.top_k = top_k

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            # if rnd < self.epsilon: # this is the correct form
            if rnd < self.epsilon:
                actions = get_random_actions(self.num_actions, self.top_k)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
            else:
                Q_a = self.actor.forward(state.unsqueeze(
                    0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                actions = get_actions(Q_a[0], self.top_k)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
        return actions, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal):
        self._step += 1
        self._add_sample(state, action, reward,
                         next_state, next_action, terminal)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        # assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(
            state, action, reward, next_state, terminal=terminal)
