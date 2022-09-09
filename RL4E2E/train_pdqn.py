import json
import logging
import os
from statistics import mode
import time
import numpy as np
import argparse
from agents.pdqn import PDQNAgent
from RL4E2E.environemnt.multiwoz_simulator import MultiwozSimulator
# from wrappers.interfaces import GalaxyInterface, PptodInterface
from tqdm import tqdm
from RL4E2E.utils.constants import FRAMEWORK_PATH


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


def train(env,args,log_path):
    logging.basicConfig(level=logging.INFO, filename=log_path)
    infos  = []
    env.seed(args.seed)
    np.random.seed(args.seed)
    assert env.num_selected_actions==args.num_selected_actions
    agent = PDQNAgent(
                    env.observation_space.shape , env.action_space,
                    k = env.num_selected_actions,
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
                    seed=args.seed,
                    log_path = log_path)
    if args.save_dir:
        save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version+"k"+str(env.num_selected_actions))
        if os.path.exists(save_dir) and len(os.listdir(save_dir))!= 0:
            checkpoints = [int(cpt.split("_")[1]) for cpt in os.listdir(save_dir) if cpt.isdigit() ]
            if len(checkpoints)!=0:
                agent.load_models( os.path.join(save_dir,"episode_"+str(max(checkpoints))) )
            else :
                checkpoints = [0]
                os.makedirs(save_dir, exist_ok=True)
        else :
            checkpoints = [0]
            os.makedirs(save_dir, exist_ok=True)
    total_reward = 0.
    returns = []
    start_time = time.time()
    pbar = tqdm(range(args.episodes-max(checkpoints)))
    try :
        for i in pbar:
            info = {}
            info['turn'] = []
            state = env.reset()
            episode_reward = 0.
            logging.info("the reward is reset to 0")
            # agent.start_episode()
            done = False
            next_action ,ac_n ,all_action_parameters  = None, None, None
            info['valids'] = '0'
            while not done:
                turn_info = {}
                # turn_info['valids'] = 0
                if next_action is None:
                    state = np.array(state, dtype=np.float32, copy=False)
                    act, action_parameters = agent.act(state)
                    action_parameters = pad_action(act, action_parameters , agent.top_k)
                    action = (act, action_parameters)
                    all_action_parameters = enhance_action(act, action_parameters , agent.top_k)
                    all_action = (act, all_action_parameters)
                else :
                    all_action = (ac_n, all_action_parameters)
                info['episode'] = int(i)
                turn_info['state'] = np.array(state, dtype=str).tolist() 
                info['dialogue'] = env.hidden_state_dial_name
                info['actions'] = np.array(all_action[0], dtype=str).tolist() 
                info['actions_params'] = np.array(all_action[1], dtype=str).tolist() 
                ret = env.step(all_action)
                utterance,new_utterance,next_state, reward, trans_rate, done, successful = ret
                next_state = np.array(next_state, dtype=np.float32, copy=False)
                next_action = agent.act(next_state)
                ac_n, p_n = next_action
                ac_, p_ = action
                all_action_parameters = enhance_action(ac_n, p_n , agent.top_k)
                all_action = (ac_n, all_action_parameters)
                agent.step(state, p_, reward, next_state, p_n, done)
                state = next_state
                turn_info['utterance'] = utterance
                turn_info['new_utterance'] = new_utterance
                turn_info['reward'] = str(reward)
                turn_info['trans_rate'] = str(trans_rate)
                turn_info['valid'] = trans_rate>= 0.25
                if turn_info['valid'] :
                    info['valids']=str(int(info['valids'])+1)
                turn_info['done'] = done
                episode_reward += reward
                info['turn'].append(turn_info)
                if done:
                    info['successful'] = successful
                    info['valid'] = bool(int(info['valids'])/state[0] > 0.5)
                    info['episode_reward'] = str(episode_reward)
                    info['avg_reward'] = str(episode_reward/len(info['turn']))
                    infos.append(info)
                    break
            agent.end_episode()
            returns.append(episode_reward)
            total_reward += episode_reward
            logging.info(f"after episode {i} total_reward is, {total_reward}")
            if i+max(checkpoints) != 0 and (i+max(checkpoints)) % args.save_freq == 0:
                os.mkdir(os.path.join(save_dir,"episode_"+str(i+max(checkpoints))))
                agent.save_models(os.path.join(save_dir,"episode_"+str(i+max(checkpoints))))
                file = open(os.path.join(save_dir,"infos_train"+str(max(checkpoints))+".json"), "w")
                json.dump(infos, file, indent=4)
                file.close()
            message = "reward after episode "+str(i+max(checkpoints))+" is "+str(total_reward)
            pbar.set_description(message)
    except KeyboardInterrupt:
        info['turn'].append(turn_info)
        infos.append(info)
        file = open(os.path.join(save_dir,"infos_train_backup.json"), "w")
        json.dump(infos, file, indent=4)
        file.close()
    except Exception:
        print(Exception)
        info['turn'].append(turn_info)
        infos.append(info)
        file = open(os.path.join(save_dir,"infos_train_backup.json"), "w")
        json.dump(infos, file, indent=4)
        file.close()
            
    except Exception:
        # logging.info("exception :",Exception)
        file = open(os.path.join(save_dir,"infos.json"), "w")
        json.dump(infos, file, indent=4)
        file.close()
    end_time = time.time()
    file = open(os.path.join(save_dir,"infos_train.json"), "w")
    json.dump(info, file, indent=4)
    file.close()
    logging.info("Took %.2f seconds" % (end_time - start_time))
    env.close()
    agent.save_models(save_dir)
    # logging.info(f"Ave. return = {sum(returns) / len(returns)}")
    np.save(save_dir,returns)  
    

def evaluate(env,args,logger):

    # if args.save_dir:
    #     save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version)
    #     os.makedirs(save_dir, exist_ok=True)
     
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
    if args.save_dir:
        save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version+"k"+str(env.num_selected_actions))
        if os.path.exists(save_dir) and len(os.listdir(save_dir))!= 0:
            try :
                checkpoints = [int(cpt.split("_")[1]) for cpt in os.listdir(save_dir) if cpt.isdigit() ]
                agent.load_models( os.path.join(save_dir,"episode_"+str(max(checkpoints))) )
            except :
                checkpoints = [0]
                agent.load_models( save_dir)
        else :
            checkpoints = [0]
            agent.load_models( save_dir)
    returns = []
    # timesteps = []
    logging.basicConfig(level=logging.INFO, filename=log_path)
    infos  = []
    total_reward = 0.
    pbar = tqdm(range(args.evaluation_episodes))
    start_time = time.time()
    try :
        for i in pbar:
            info = {}
            info['turn'] = []
            info['valids'] = '0'
            state = env.reset()
            episode_reward = 0.
            done = False
            # next_action ,ac_n ,all_action_parameters  = None, None, None
            while not done:
                turn_info = {}
                # if next_action is None:
                state = np.array(state, dtype=np.float32, copy=False)
                act, action_parameters = agent.act(state)
                action_parameters = pad_action(act, action_parameters , agent.top_k)
                action = (act, action_parameters)
                all_action_parameters = enhance_action(act, action_parameters , agent.top_k)
                all_action = (act, all_action_parameters)
                # else :
                #     all_action = (ac_n, all_action_parameters)
                info['episode'] = int(i)
                turn_info['state'] = np.array(state, dtype=str).tolist() 
                info['dialogue'] = env.hidden_state_dial_name
                info['actions'] = np.array(all_action[0], dtype=str).tolist() 
                info['actions_params'] = np.array(all_action[1], dtype=str).tolist() 
                ret = env.step(all_action)
                utterance,new_utterance,next_state, reward, trans_rate, done, successful = ret
                turn_info['utterance'] = utterance
                turn_info['new_utterance'] = new_utterance
                turn_info['reward'] = str(reward)
                turn_info['trans_rate'] = str(trans_rate)
                turn_info['valid'] = trans_rate>= 0.25
                if turn_info['valid'] :
                    info['valids']=str(int(info['valids'])+1)
                turn_info['done'] = done
                episode_reward += reward
                info['turn'].append(turn_info)
                state = next_state
                if done:
                    info['successful'] = successful
                    info['valid'] = bool(int(info['valids'])/state[0] > 0.5)
                    info['episode_reward'] = str(episode_reward)
                    info['avg_reward'] = str(episode_reward/len(info['turn']))
                    infos.append(info)
                    break
                
                
            agent.end_episode()
            returns.append(episode_reward)
            total_reward += episode_reward
            logging.info(f"after episode {i} total_reward is, {total_reward}")
            if i+max(checkpoints) != 0 and i+max(checkpoints) % args.save_freq == 0:
                os.mkdir(os.path.join(save_dir,"episode_"+str(i+max(checkpoints))))
                agent.save_models(os.path.join(save_dir,"episode_"+str(i+max(checkpoints))))
            message = "reward after episode "+str(i+max(checkpoints))+" is "+str(total_reward)
            pbar.set_description(message)
    except KeyboardInterrupt:
        info['turn'].append(turn_info)
        infos.append(info)
        file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
        json.dump(infos, file, indent=4)   
        file.close()   
    except :
        info['turn'].append(turn_info)
        infos.append(info)
        file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
        json.dump(infos, file, indent=4)   
        file.close()  
        
    end_time = time.time()
    file = open(os.path.join(save_dir,"infos_eval.json"), "w")
    json.dump(infos, file, indent=4)
    file.close()
    logging.info("Took %.2f seconds" % (end_time - start_time))
    env.close()


def main(env,args,log_path):
    if args.action=="train":
        train(env,args,log_path)
    else :
        evaluate(env,args,log_path)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', default=500, help='Episodes over which to evaluate after training.', type=int) # episodes = 1000
    parser.add_argument('--episodes', default=10000, help='Number of epsiodes.', type=int) #20000
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
    parser.add_argument('--save_freq', default=500, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save_dir', default=os.path.join(FRAMEWORK_PATH,"results"), help='Output directory.', type=str)
    parser.add_argument('--action', default="train", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    parser.add_argument('--version', default="2.0", choices=["2.0", "2.1"], help="the multiwoz version we want to use", type=str) 
    parser.add_argument('--num_selected_actions', default=1, help="how many actions to apply simultaniously", type=int) 
    args = parser.parse_args()
    mode = "dev" if args.action == 'train' else "test"
    logging_path = os.path.join(FRAMEWORK_PATH,"results/"+args.model+"/"+args.version+'/'+str(args.num_selected_actions))
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    log_path = os.path.join(logging_path,"output.log")
    env = MultiwozSimulator(model=args.model, version=args.version, num_selected_actions=args.num_selected_actions, mode=mode , log_path = log_path)
    main(env,args,log_path)