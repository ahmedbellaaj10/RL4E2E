import argparse
import json
import logging
import os
import random
import time
import numpy as np
from gym import spaces
from tqdm import tqdm
from RL4E2E.environemnt.multiwoz_simulator import MultiwozSimulator
from RL4E2E.transformations.constants import CHAR_DROP_VECTOR_SIZE, CHAR_INSERT_VECTOR_SIZE, CHAR_REPLACE_VECTOR_SIZE, VALID_RATE , MAX_WORDS, WORD_DROP_VECTOR_SIZE, WORD_INSERT_VECTOR_SIZE, WORD_REPLACE_VECTOR_SIZE 
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

def get_random_actions(k):
    action_ids = [0,1,2,3,4,5]
    probability = np.random.random(k)
    probability /= probability.sum()
    actions = random.sample(action_ids, k)
    actions.sort()
    vect = []
    i =0
    for act in action_ids:
        vect.append(act)
        if act in actions:
            vect.append(probability[i])
            i+=1
        else :
            vect.append(0.0)
    return vect

def run(env,args,log_path):
    logging.basicConfig(level=logging.INFO, filename=log_path)
    infos  = []
    # try :
    env.seed(args.seed)
    np.random.seed(args.seed)
    assert env.num_selected_actions==args.num_selected_actions
    if args.save_dir:
        save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version+"k"+str(env.num_selected_actions))
        os.makedirs(save_dir, exist_ok=True)
    returns = []
    start_time = time.time()
    pbar = tqdm(range(args.evaluation_episodes))
    max_params = max(  WORD_INSERT_VECTOR_SIZE,
                                WORD_DROP_VECTOR_SIZE,
                                WORD_REPLACE_VECTOR_SIZE ,
                                CHAR_INSERT_VECTOR_SIZE ,
                                CHAR_DROP_VECTOR_SIZE , 
                                CHAR_REPLACE_VECTOR_SIZE)
    params = spaces.Tuple(  # parameters
                tuple(spaces.Box(low=np.zeros(int(VALID_RATE * MAX_WORDS * max_params),), high=np.ones(int(VALID_RATE * MAX_WORDS* max_params),), dtype=np.float32)
                      for i in range(int(args.num_selected_actions)))
            )
    try :
        for i in pbar:
            info = {}
            info['turn'] = []
            info['valids'] = '0'
            state = env.reset()
            done = False
            cumulate = True # random.choice([True , False])
            while not done :
                turn_info = {}
                # if next_action is None:
                state = np.array(state, dtype=np.float32, copy=False)
                actions = get_random_actions(args.num_selected_actions)
                action_params = params.sample()
                action_parameters = pad_action(actions, action_params , args.num_selected_actions)
                all_action_parameters = enhance_action(actions, action_parameters , args.num_selected_actions)
                all_action = (actions, all_action_parameters)
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
                info['turn'].append(turn_info)
                
                if done:
                    info['successful'] = successful
                    info['valid'] = bool(int(info['valids'])/state[1] > 0.5)
                    infos.append(info)
                    break
                state = next_state
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
        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=512, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', default=300, help='Episodes over which to evaluate after training.', type=int) # episodes = 1000
    parser.add_argument('--save_dir', default=os.path.join(FRAMEWORK_PATH,"results/random_sampler"), help='Output directory.', type=str)
    # parser.add_argument('--render_freq', default=100, help='How often to render / save frames of an episode.', type=int)
    # parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)
    parser.add_argument('--action', default="test", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    parser.add_argument('--version', default="2.1", choices=["2.0", "2.1"], help="the multiwoz version we want to use", type=str) 
    parser.add_argument('--num_selected_actions', default=1, help="how many actions to apply simultaniously", type=int) 
    args = parser.parse_args()
    mode = "test"
    logging_path = os.path.join(FRAMEWORK_PATH,"results/random_sampler/"+args.model+"/"+args.version)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    log_path = os.path.join(logging_path,"rs_output.log")
    env = MultiwozSimulator(model=args.model, version=args.version, num_selected_actions=args.num_selected_actions, mode=mode , log_path = log_path)
    run(env,args,log_path)