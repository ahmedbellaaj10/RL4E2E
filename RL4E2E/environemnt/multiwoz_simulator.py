from asyncio.log import logger
import random
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Discrete
from gym.utils import seeding
import numpy as np
import torch
import sys
from RL4E2E.transformations.transformer import CharReplace, CharInsert, CharDrop, WordDrop, WordInsert, WordReplace, CompoundTransformer
from RL4E2E.transformations.constants import CHAR_DROP_VECTOR_SIZE, CHAR_INSERT_VECTOR_SIZE, CHAR_REPLACE_VECTOR_SIZE, VALID_RATE , MAX_WORDS, WORD_DROP_VECTOR_SIZE, WORD_INSERT_VECTOR_SIZE, WORD_REPLACE_VECTOR_SIZE , TRANSFORMATIONS
from wrappers.interfaces import GalaxyInterface, PptodInterface
import logging
sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")

logging.basicConfig(filename='output.log' , level=logging.INFO ,format="%(message)s")

ACTIONS = {
    0: WordDrop(),
    1: WordInsert(),
    2: WordReplace(),
    3: CharDrop(),
    4: CharInsert(),
    5: CharReplace()
}


class MultiwozSimulator(gym.Env):

    def __init__(self , dataset="multiwoz", version="2.0" , model="galaxy", num_selected_actions=3, mode="dev" ):
        self.mode = mode
        self.num_selected_actions = num_selected_actions 
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
        except :
            print(f"Be careful, the model {self.model} is not supported for the moment")

        if self.model.lower() == "galaxy":
            self.interface = GalaxyInterface()
        elif self.model.lower() == "pptod":
            self.interface = PptodInterface()

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
        (actions, all_params) = action 
        logging.info(f"the actions vector is {actions}")
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
        logging.info(f"before transformation, the delexicalized sentence was: {utterance_delex}")
        utterance , utterance_delex , idxs = self.remove_keywords(utterance , utterance_delex)
        new_utterance = self.compound_transfomer.apply(utterance, action)
        new_utterance_delex = self.compound_transfomer.apply(utterance_delex , action)
        utterance , utterance_delex = self.restore_keywords(utterance , utterance_delex , idxs)
        logging.info(f"after transformation, the sentence was: {utterance}")
        logging.info(f"after transformation, the delexicalized sentence was: {utterance_delex}")
        turn_modified = self.interface.copy_dial_or_turn(turn)
        self.interface.set_utterance_and_utterance_delex(turn_modified, num_current_turn ,new_utterance, new_utterance_delex)
        turn_encoded , turn_mdified_encoded = self.interface.encode_turn(dialogue_name , turn) , self.interface.encode_turn(dialogue_name , turn_modified)
        turn_predict, _, _ = self.interface.predict_turn(turn_encoded)
        turn_modified_predict, _, _ = self.interface.predict_turn(turn_mdified_encoded)
        bleu1 ,  success1 , match1 = self.interface.evaluate(turn_predict)
        bleu2 , success2 , match2 = self.interface.evaluate(turn_modified_predict)
        logging.info(f"bleu score for real data was : {bleu1}")
        logging.info(f"bleu score for transformed data was : {bleu2}")
        reward = 0
        # if cumulate :
        self.interface.set_utterance_and_utterance_delex(dialogue_copy, num_current_turn ,new_utterance, new_utterance_delex)
        # else leave it as it is
        self.state[self.Current_Turn_Order]+=1
        self.state[self.Remaining_Turns_Order]-=1
        if self.state[self.Remaining_Turns_Order]==0 :
            done = True
            logger.debug("this dialogue is done")
            # encoded = self.interface.encode_dialogue(dialogue_name, dialogue_copy)
            # print("dialogue_copy",dialogue_copy)
            # print("encoded",encoded)
            # results , bleu , success , match = self.interface.predict_dialogue(encoded)
            if round(success2 + match2) == 200:
                reward += 1
            self.state = self.reset()
        # if bleu1 - bleu2 <= 0 :
        #     reward += min(-1, (bleu1 - bleu2)/100 )
        # else :
        reward += (bleu1 - bleu2)/100
        logger.debug("reward is {reward}" )
        return self.state, reward, done, {}
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def remove_keywords(self , utterance , utterance_delex):
        user = utterance.split()
        delex = utterance_delex.split()
        idxs = {}
        for idx , word in enumerate(delex) :
            if word.startswith('[') and word.endswith(']'):
                idxs.update({idx : [user[idx] , word]})
                delex.remove(word)
                user.remove(user[idx])
        utterance = ' '.join(user)
        utterance_delex = ' '.join(delex)
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
# if __name__ == "__main__":
#     interface = "galaxy"
#     if interface == "galaxy" :
#         x = GalaxyInterface()
#         dial = x.get_dialogue("sng0073")
#         print("len dial", len(dial))
#         turn = x.get_turn_with_context(dial , 1)
#         print("turn",turn)
#         encoded = x.encode("sng0073",turn)
#         print("encode",encoded)

#         turn_output, bleu , tmp_dialog_result , pv_turn = x.predict_turn(encoded, 1)
#         print("----bleu", bleu)
#         print("---------------------------------")
#         print("----turn_output", turn_output)
#         print("---------------------------------")
#         print("----tmp_dialog_result", tmp_dialog_result)
#         print("---------------------------------")
#         print("----pv_turn", pv_turn)
#         print("---------------------------------")
#     else :
#         with torch.no_grad():
#             x = PptodInterface()
#             data = x.data
#             dialogue = x.get_dialogue("sng0073" , mode = 'dev')
#             print("get dialogue done")
#             turn = x.get_turn_with_context(dialogue , 1)
#             print("turn with context", turn)
#             # turn = x.get_turn(dialogue , 1)
#             # print("turn", turn)
#             print("get turn done")
            
#             # input("fdbdsfbdsgbdgs")
#             prepared_dial = x.prepare_turn(turn)
#             print("preparing data done")
#             print("prepared dial", prepared_dial)
#             # import time
#             # time.sleep(10)
#             dial_result = x.predict_turn(prepared_dial)
#             print("dial result", dial_result)
#             dev_bleu, dev_success, dev_match = x.evaluate_turn(dial_result)
#             print("dev_bleu",dev_bleu)
#             print("dev_success",dev_success) 
#             print("dev_match",dev_match)  