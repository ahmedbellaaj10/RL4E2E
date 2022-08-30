import argparse
import logging
import os
import time
import numpy as np
from tqdm import tqdm
from RL4E2E.environemnt.multiwoz_simulator import MultiwozSimulator
from RL4E2E.transformations.constants import CHAR_DROP_VECTOR_SIZE, CHAR_INSERT_VECTOR_SIZE, CHAR_REPLACE_VECTOR_SIZE, VALID_RATE , MAX_WORDS, WORD_DROP_VECTOR_SIZE, WORD_INSERT_VECTOR_SIZE, WORD_REPLACE_VECTOR_SIZE , TRANSFORMATIONS
from RL4E2E.transformations.transformer import CharDrop, CharInsert, CharReplace, CompoundTransformer, WordDrop, WordInsert, WordReplace
from RL4E2E.utils.constants import FRAMEWORK_PATH 

def train(env,args,log_path):
    logging.basicConfig(level=logging.INFO, filename=log_path)
    infos  = []
    # try :
    env.seed(args.seed)
    np.random.seed(args.seed)
    assert env.num_selected_actions==args.num_selected_actions
    if args.save_dir:
        save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version+"k"+str(env.num_selected_actions))
        os.makedirs(save_dir, exist_ok=True)
    total_reward = 0.
    returns = []
    start_time = time.time()
    pbar = tqdm(range(args.episodes))
    for i in pbar:
        info = {}
        info['turn'] = []
        state = env.reset()
        done = False
        cumulate = True # random.choice([True , False])
        dialogue_name , dialogue , dialogue_copy = env.hidden_state_dial_name , env.hidden_state_dialogue , env.hidden_state_dialogue_copy
        num_current_turn = env.state[env.Current_Turn_Order]
        turn = env.interface.get_turn_with_context(dialogue_copy , num_current_turn)
        utterance , utterance_delex = env.interface.get_utterance_and_utterance_delex(dialogue_copy , num_current_turn)
        utterance , utterance_delex , idxs = env.remove_keywords(utterance , utterance_delex)
        print("state",state)
        print("hidden_state", env.hidden_state_dialogue)
        params = []
        for transformation_name in TRANSFORMATIONS:
            if transformation_name.lower() == 'charinsert':
                transformation = CharInsert()

            elif transformation_name.lower() == 'chardrop':
                transformation = CharDrop()

            if transformation_name.lower() == 'charreplace':
                transformation = CharReplace()

            elif transformation_name.lower() == 'worddrop':
                transformation = WordDrop()

            if transformation_name.lower() == 'wordinsert':
                transformation = WordInsert()

            elif transformation_name.lower() == 'wordreplace':
                transformation = WordReplace()
            # for x in range(int(VALID_RATE*MAX_WORDS)):
            params+= transformation.sample(utterance)
        compound_transfomer = CompoundTransformer()
        CompoundTransformer.apply(utterance, params)
        new_utterance_delex , _ = compound_transfomer.apply(utterance_delex , params)
        new_utterance , new_utterance_delex = env.restore_keywords(new_utterance , new_utterance_delex , idxs)

        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=512, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', default=10, help='Episodes over which to evaluate after training.', type=int) # episodes = 1000
    parser.add_argument('--episodes', default=1, help='Number of epsiodes.', type=int) #20000
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
    parser.add_argument('--save_freq', default=200, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save_dir', default=os.path.join(FRAMEWORK_PATH,"results"), help='Output directory.', type=str)
    # parser.add_argument('--render_freq', default=100, help='How often to render / save frames of an episode.', type=int)
    # parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)
    parser.add_argument('--action', default="train", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    parser.add_argument('--version', default="2.0", choices=["2.0", "2.1"], help="the multiwoz version we want to use", type=str) 
    parser.add_argument('--num_selected_actions', default=1, help="how many actions to apply simultaniously", type=int) 
    args = parser.parse_args()
    mode = "dev" if args.action == 'train' else "test"
    logging_path = os.path.join(FRAMEWORK_PATH,"results/"+args.model+"/"+args.version)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    log_path = os.path.join(logging_path,"output.log")
    env = MultiwozSimulator(model=args.model, version=args.version, num_selected_actions=args.num_selected_actions, mode=mode , log_path = log_path)
    train(env,args,log_path)