import math
import random
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete
from gym.utils import seeding
import numpy as np
import sys
from RL4E2E.transformations.transformer import CharReplace, CharInsert, CharDrop, WordDrop, WordInsert, WordReplace, CompoundTransformer
from RL4E2E.transformations.constants import CHAR_DROP_VECTOR_SIZE, CHAR_INSERT_VECTOR_SIZE, CHAR_REPLACE_VECTOR_SIZE, VALID_RATE , MAX_WORDS, WORD_DROP_VECTOR_SIZE, WORD_INSERT_VECTOR_SIZE, WORD_REPLACE_VECTOR_SIZE , TRANSFORMATIONS
from RL4E2E.utils.constants import PPTOD_PATH,  GALAXY_PATH ,FRAMEWORK_PATH
from wrappers.interfaces import GalaxyInterface, PptodInterface
import logging
sys.path.append(GALAXY_PATH)
sys.path.append(PPTOD_PATH)

from RL4E2E.utils.scores import bleu 

ACTIONS = {
    0: "WordInsert",
    1: "WordDrop",
    2: "WordReplace",
    3: "CharInsert",
    4: "CharDrop",
    5: "CharReplace"
}


class MultiwozSimulator(gym.Env):

    def __init__(self , dataset="multiwoz", version="2.0" , model="galaxy", num_selected_actions=3, mode="dev", log_path = ''):
        self.mode = mode
        self.num_selected_actions = num_selected_actions 
        logging.basicConfig(level=logging.INFO, filename=log_path)
        try :
            self.dataset = dataset
            assert self.dataset.lower() == "multiwoz"
        except :
            print(f"Be careful, {self.dataset} dataset is not supported for the moment")
        try :
            self.version = version
            assert self.version in ["2.0" , "2.1"]
        except :
            print(f"Be careful, version {self.version} is not supported for the moment")
        try :
            self.model = model
            assert self.model.lower() in ["galaxy" , "pptod"]
            print("model is", self.model)
        except :
            print(f"Be careful, the model {self.model} is not supported for the moment")

        if self.model.lower() == "galaxy":
            self.interface = GalaxyInterface(self.version, log_path)
        elif self.model.lower() == "pptod":
            self.interface = PptodInterface(self.version, log_path)

        # path = os.path.join(FRAMEWORK_PATH , os.path.join(self.model.lower(),self.version))
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # logging.basicConfig(filename=os.path.join(path,'output.log')  , level=logging.INFO ,format="%(message)s")
        self.compound_transfomer = CompoundTransformer(TRANSFORMATIONS)
        self.ACTIONS = self.compound_transfomer.get_actions()
        self.Dialogue_Idx_Order = 0
        self.Current_Turn_Order = 1
        self.Remaining_Turns_Order = 2
        self.STATE_ELEMENTS = 3
        self.hidden_state_dial_name , self.hidden_state_dialogue = "" , None

        self.num_possible_dialogues = len(self.interface.dev_data) if (self.mode == "dev") else len(self.interface.test_data)

        self.state = [None]*self.STATE_ELEMENTS
        self.observation_space = MultiDiscrete(np.array([self.num_possible_dialogues , 20 , 20]))
        self.num_actions = len(list(ACTIONS.keys()))
        # self.action_space = Discrete(10) #temporarely
        self.max_params = max(  WORD_INSERT_VECTOR_SIZE,
                                WORD_DROP_VECTOR_SIZE,
                                WORD_REPLACE_VECTOR_SIZE ,
                                CHAR_INSERT_VECTOR_SIZE ,
                                CHAR_DROP_VECTOR_SIZE , 
                                CHAR_REPLACE_VECTOR_SIZE)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions),
            *spaces.Tuple(  # parameters
                tuple(spaces.Box(low=np.zeros(int(VALID_RATE * MAX_WORDS * (self.max_params)),), high=np.ones(int(VALID_RATE * MAX_WORDS* (self.max_params)),), dtype=np.float32)
                      for i in range(int(self.num_selected_actions)))
            )
        ))
        self.state = self.reset()

    def reset(self):
        logging.info("Getting new dialogue")
        self.state= [None,None,None]
        # test = False
        # while not test :
        self.state[self.Dialogue_Idx_Order] , self.hidden_state_dial_name , self.hidden_state_dialogue = self.interface.get_dialogue()
        self.state[self.Remaining_Turns_Order] = self.interface.get_dialogue_length(self.hidden_state_dialogue)
        logging.info(f"the choosen dialogue is {self.hidden_state_dial_name}")
        logging.info(f"it has { self.interface.get_dialogue_length(self.hidden_state_dialogue) } turns in total")
        self.state[self.Current_Turn_Order] = 0
        self.hidden_state_dialogue_copy = self.interface.copy_dial_or_turn(self.hidden_state_dialogue)
            # if self.hidden_state_dial_name == "pmul0012":
            #     test = True
        logging.info("got new dialogue")
        return self.state

    def step(self, action):
        reward = 0
        successful = False
        (actions, all_params) = action 
        logging.info(f"the actions vector is {actions}")
        for i,k in zip(actions[0::2], actions[1::2]):
            if k != 0.0:
                logging.info(f"the action {TRANSFORMATIONS[i]} is choosen")
        all_params = np.clip(
            all_params, a_min=np.zeros(len(all_params), dtype="float"), a_max=np.ones(len(all_params), dtype = "float"))
        logging.info(f"the params vector is {all_params}")
        action = (actions, all_params)
        done = False
        cumulate = True # random.choice([True , False])
        dialogue_name , dialogue , dialogue_copy = self.hidden_state_dial_name , self.hidden_state_dialogue , self.hidden_state_dialogue_copy
        num_current_turn = self.state[self.Current_Turn_Order]
        logging.info(f"currently we are at turn {num_current_turn}")
        # turn = self.interface.get_turn_with_context(dialogue , num_current_turn)
        turn = self.interface.get_turn_with_context(dialogue_copy , num_current_turn)
        utterance , utterance_delex = self.interface.get_utterance_and_utterance_delex(dialogue_copy , num_current_turn)
        logging.info(f"before transformation, the sentence was: {utterance}")
        # logging.info(f"before transformation, the delexicalized sentence was: {utterance_delex}")
        utterance_ , utterance_delex_ , idxs = self.remove_keywords(utterance , utterance_delex)
        new_utterance , trans_rate = self.compound_transfomer.apply(utterance_, action)
        # new_utterance_delex , _ = self.compound_transfomer.apply(utterance_delex , action)
        new_utterance_delex = new_utterance
        new_utterance , new_utterance_delex = self.restore_keywords(new_utterance , new_utterance_delex , idxs)
        trans_rate = (trans_rate / max(len(utterance.split()), len(new_utterance.split()))) / math.ceil(self.num_selected_actions/2)
        pen = random.uniform(0,0.1/(7-self.num_selected_actions))
        trans_rate += pen
        logging.info(f"after transformation, the sentence was: {new_utterance}")
        trans_rate = min(trans_rate , 1)
        # logging.info(f"after transformation, the delexicalized sentence was: {utterance_delex}")
        turn_modified = self.interface.copy_dial_or_turn(turn)
        self.interface.set_utterance_and_utterance_delex(turn_modified, num_current_turn ,new_utterance, new_utterance_delex)
        turn_encoded , turn_mdified_encoded = self.interface.encode_turn(dialogue_name , turn) , self.interface.encode_turn(dialogue_name , turn_modified)
        turn_predict, _, _ = self.interface.predict_turn(turn_encoded)
        resp , resp_gen = self.get_resp_and_resp_gen(turn_predict)
        turn_modified_predict, _, _ = self.interface.predict_turn(turn_mdified_encoded)
        resp , resp_gen_modified = self.get_resp_and_resp_gen(turn_modified_predict)
        bleu1 = bleu(resp , resp_gen)
        bleu2 = bleu(resp , resp_gen_modified)
        beta = 0 if trans_rate<0.25 else -101
        logging.info(f"bleu score for real data was : {bleu1}")
        logging.info(f"bleu score for transformed data was : {bleu2}")
        
        # if cumulate :
        self.interface.set_utterance_and_utterance_delex(dialogue_copy, num_current_turn ,new_utterance, new_utterance_delex)
        # else leave it as it is
        self.state[self.Current_Turn_Order]+=1
        self.state[self.Remaining_Turns_Order]-=1
        if self.state[self.Remaining_Turns_Order]==0 :
            done = True
            logging.info("this dialogue is done")
            bleu_sc , success , match = self.interface.evaluate(turn_modified_predict)
            if round(success + match) == 200:
                successful = True
                reward += 10
            self.state = self.reset()
        bleu_diff = bleu1 - bleu2 if bleu1 - bleu2>0 else (bleu1 - bleu2) -100
        x = (bleu_diff) + trans_rate*(beta+1)
        # if trans_rate else (bleu_diff)
        reward += x
        logging.info(f"reward is {reward}")
        return utterance,new_utterance, self.state, reward, trans_rate, done, successful
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def remove_keywords(self , utterance , utterance_delex):
        user = utterance.split()
        delex = utterance_delex.split()
        idxs = {}
        i = 0
        for idx , word in enumerate(delex):
            if word.startswith('[') and word.endswith(']') or word.startswith('<') and word.endswith('>'):
                try :
                    kw = user[idx-i]
                    user.remove(user[idx-i])
                    try :
                        while user[idx] != delex[idx+1]:
                            kw = kw+" "+user[idx]
                            user.remove(user[idx])
                        idxs.update({idx+i : [kw , word]})
                        delex.remove(word)
                        i+=1
                    except :
                        pass
                except  :
                    pass

                
                
        utterance = ' '.join(user)
        utterance_delex = ' '.join(delex)
        print("utterance",utterance)
        print("utterance_delex",utterance_delex)
        return utterance , utterance_delex , idxs

    def restore_keywords(self ,utterance , utterance_delex , idxs):
        user = utterance.split(' ')
        delex = utterance_delex.split(' ')
        for key, value in idxs.items():
            user.insert(key, value[0])
            delex.insert(key, value[1])
        utterance = ' '.join(user)
        utterance_delex = ' '.join(delex)
        return utterance , utterance_delex 

    def get_resp_and_resp_gen(self , pred):
        return pred[0]['resp'] , pred[0]['resp_gen']

