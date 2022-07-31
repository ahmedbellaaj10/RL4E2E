import os
import click
import time
# import gym
import numpy as np
import argparse
import copy
from agents.pdqn import PDQNAgent
from RL4E2E.environemnt.multiwoz_simulator import MultiwozSimulator
from wrappers.interfaces import GalaxyInterface, PptodInterface
from tqdm import tqdm


def pad_action(acts, act_param, k):
    n = (len(act_param)//k)
    has_params = 0
    params = []
    log = []
    for idx , act in enumerate(acts):
        if isinstance(act, int) and acts[idx+1] != 0.0:
            vals = {}
            vals['action']=acts[idx]
            vals['ratio']=acts[idx+1]
            vals['params'] = np.array(act_param[has_params*n:n*(has_params+1)])
            has_params+=1
            params.append(vals['params'])
            log.append(vals)
        else :
            pass 

    x = np.reshape(params, (-1,))
    # print(x.shape)
    return np.reshape(params, (-1,))

def enhance_action(acts, act_param, k):
    n = (len(act_param)//k)
    has_params = 0
    params = []
    for idx , act in enumerate(acts):
        if isinstance(act, int) and isinstance(acts[idx+1], float) and acts[idx+1] != 0.0:
            vals = {}
            vals['action']=acts[idx]
            vals['ratio']=acts[idx+1]
            vals['params'] = np.array(act_param[has_params*n:n*(has_params+1)])
            has_params+=1
            params.append(vals['params'])
        elif isinstance(act, int) and isinstance(acts[idx+1], float) and acts[idx+1] == 0.0:
            vals = {}
            vals['action']=acts[idx]
            vals['ratio']=acts[idx+1]
            vals['params'] = np.zeros(n)
            params.append(vals['params'])
        else :
            pass 
    return np.reshape(params, (-1,))


def train(env,args):
    interface = PptodInterface()
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
        os.makedirs(args.save_dir, exist_ok=True)
    env.seed(args.seed)
    np.random.seed(args.seed)
    agent = PDQNAgent(
                       env.observation_space.shape , env.action_space,
                       env.num_selected_actions,
                       batch_size=args.batch_size,
                       learning_rate_actor=args.learning_rate_actor,
                       learning_rate_actor_param=args.learning_rate_actor_param,
                       epsilon_steps=args.epsilon_steps,
                       gamma=args.gamma,
                       tau_actor=args.tau_actor,
                       tau_actor_param=args.tau_actor_param,
                       clip_grad=args.clip_grad,
                       indexed=args.indexed,
                       weighted=args.weighted,
                       average=args.average,
                       random_weighted=args.random_weighted,
                       initial_memory_threshold=args.initial_memory_threshold,
                       use_ornstein_noise=args.use_ornstein_noise,
                       replay_memory_size=args.replay_memory_size,
                       epsilon_final=args.epsilon_final,
                       inverting_gradients=args.inverting_gradients,
                       actor_kwargs={'hidden_layers': args.layers,
                                     'action_input_layer': args.action_input_layer,},
                       actor_param_kwargs={'hidden_layers': args.layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=args.zero_index_gradients,
                       seed=args.seed)
    total_reward = 0.
    returns = []
    start_time = time.time()
    for i in tqdm(range(args.episodes)):
        print("new episode")
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, action_parameters = agent.act(state)
        action_parameters = pad_action(act, action_parameters , agent.k)
        action = (act, action_parameters)
        all_action_parameters = enhance_action(act, action_parameters , agent.k)
        all_action = (act, all_action_parameters)
        episode_reward = 0.
        agent.start_episode()
        done = False
        while not done:
            ret = env.step(all_action)
            next_state, reward, done, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_action = agent.act(next_state)
            ac_n, p_n = next_action
            ac_, p_ = action
            all_action_parameters = enhance_action(ac_n, p_n , agent.k)
            all_action = (ac_n, all_action_parameters)
            agent.step(state, p_, reward, next_state, p_n, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        agent.end_episode()
        returns.append(episode_reward)
        total_reward += episode_reward
        print(f"after episode {i} total_reward is, {total_reward}")
        # if i % args.disp_freq == 0:
        # print('{0:5s} R:{1:.4f} r:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-args.disp_freq:]).mean()))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    agent.save_models(args.save_dir)
    print("Ave. return =", sum(returns) / len(returns))
    np.save(os.path.join(dir, args.title + "{}".format(str(args.seed))),returns)  
    

def evaluate(env,args):
    if args.save_dir:
        save_dir = os.path.join(args.save_dir, args.title + "{}".format(str(args.seed)))
        os.makedirs(args.save_dir, exist_ok=True)
     
    agent = PDQNAgent(
                       env.observation_space.shape, env.action_space,
                       batch_size=args.batch_size,
                       learning_rate_actor=args.learning_rate_actor,
                       learning_rate_actor_param=args.learning_rate_actor_param,
                       epsilon_steps=args.epsilon_steps,
                       gamma=args.gamma,
                       tau_actor=args.tau_actor,
                       tau_actor_param=args.tau_actor_param,
                       clip_grad=args.clip_grad,
                       indexed=args.indexed,
                       weighted=args.weighted,
                       average=args.average,
                       random_weighted=args.random_weighted,
                       initial_memory_threshold=args.initial_memory_threshold,
                       use_ornstein_noise=args.use_ornstein_noise,
                       replay_memory_size=args.replay_memory_size,
                       epsilon_final=args.epsilon_final,
                       inverting_gradients=args.inverting_gradients,
                       actor_kwargs={'hidden_layers': args.layers,
                                     'action_input_layer': args.action_input_layer,},
                       actor_param_kwargs={'hidden_layers': args.layers,
                                           'squashing_function': False,
                                           'output_layer_init_std': 0.0001,},
                       zero_index_gradients=args.zero_index_gradients,
                       seed=args.seed)
    print('in evaluate savedir is :', args.save_dir)
    agent.load_models(os.path.join(args.save_dir,args.title+"1"))
    returns = []
    timesteps = []
    for _ in range(args.evaluation_episodes):
        state = env.reset()
        done = False
        t = 0
        total_reward = 0.
        while not done:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, all_action_parameters, _ = agent.act(state)
            action = (act, all_action_parameters)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    return np.array(returns)


def main(env,args):
    if args.action=="train":
        train(env,args)
    evaluate(env,args)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=12, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', default=10, help='Episodes over which to evaluate after training.', type=int) # episodes = 1000
    parser.add_argument('--episodes', default=300, help='Number of epsiodes.', type=int) #20000
    parser.add_argument('--batch_size', default=128, help='Minibatch size.', type=int)
    parser.add_argument('--gamma', default=0.9, help='Discount factor.', type=float)
    parser.add_argument('--inverting_gradients', default=True,
                help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
                type=int)  # may have been running with 500??
    parser.add_argument('--use_ornstein_noise', default=True,
                help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--replay_memory_size', default=1000, help='Replay memory size in transitions.', type=int)
    parser.add_argument('--epsilon_steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
    parser.add_argument('--epsilon_final', default=0.01, help='Final epsilon value.', type=float)
    parser.add_argument('--tau_actor', default=0.1, help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor_param', default=0.001, help='Soft target network update averaging factor.', type=float) 
    parser.add_argument('--learning_rate_actor', default=0.001, help="Actor network learning rate.", type=float)
    parser.add_argument('--learning_rate_actor_param', default=0.0001, help="Critic network learning rate.", type=float)  
    parser.add_argument('--initialise_params', default=True, help='Initialise action parameters.', type=bool)
    parser.add_argument('--clip_grad', default=10., help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--indexed', default=False, help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False, help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False, help='Average weighted loss function.', type=bool)
    parser.add_argument('--random_weighted', default=False, help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--zero_index_gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
    parser.add_argument('--action_input_layer', default=0, help='Which layer to input action parameters.', type=int)
    parser.add_argument('--layers', default=(128,), help='Duplicate action-parameter inputs.')
    # parser.add_argument('--save_freq', default=0, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save_dir', default="/home/ahmed/RL4E2E/results", help='Output directory.', type=str)
    # parser.add_argument('--render_freq', default=100, help='How often to render / save frames of an episode.', type=int)
    parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)
    parser.add_argument('--action', default="train", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    args = parser.parse_args()
    env = MultiwozSimulator(model=args.model)
    print(args)
    main(env,args)