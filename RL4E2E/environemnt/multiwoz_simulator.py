import gym
from gym.spaces import MultiDiscrete, Discrete
import numpy as np
import torch
import sys
from wrappers.interfaces import GalaxyInterface, PptodInterface
sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")

class MultiwozSimulator(gym.Env):
    ENV_CONSTANTS = {

    }

    def __init__(self , dataset, version , model):
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

        self.Dialogue_Idx_Order = 0
        self.Current_Turn_Order = 1
        self.Remaining_Turns_Order = 2
        self.STATE_ELEMENTS = 3
        self.hidden_state = ""

        self.num_possible_dialogues = 10 # normalemnt ca dépend men kol naw3 mte3 data :


        self.state = [None]*self.STATE_ELEMENTS
        self.observation_space = MultiDiscrete(np.array([self.num_possible_dialogues , 20 , 20]))

        self.action_space = Discrete(10) #normally it should be more than that

    def reset(self):
        state= [None,None,None]
        state[self.Dialogue_Idx_Order] , self.hidden_state , state[self.Remaining_Turns_Order] = self.interface.choose_dialogue()
        state[self.Current_Turn_Order] = 0
        return state

    def step(self, action):
        pass

if __name__ == "__main__":
    interface = "galaxy"
    if interface == "galaxy" :
        x = GalaxyInterface()
        dial = x.get_dialogue("sng0073")
        print("len dial", len(dial))
        turn = x.get_turn_with_context(dial , 1)
        print("turn",turn)
        encoded = x.encode("sng0073",turn)
        print("encode",encoded)

        turn_output, bleu , tmp_dialog_result , pv_turn = x.predict_turn(encoded, 1)
        print("----bleu", bleu)
        print("---------------------------------")
        print("----turn_output", turn_output)
        print("---------------------------------")
        print("----tmp_dialog_result", tmp_dialog_result)
        print("---------------------------------")
        print("----pv_turn", pv_turn)
        print("---------------------------------")
    else :
        with torch.no_grad():
            x = PptodInterface()
            data = x.data
            dialogue = x.get_dialogue("sng0073" , mode = 'dev')
            print("get dialogue done")
            turn = x.get_turn_with_context(dialogue , 1)
            print("turn with context", turn)
            # turn = x.get_turn(dialogue , 1)
            # print("turn", turn)
            print("get turn done")
            
            # input("fdbdsfbdsgbdgs")
            prepared_dial = x.prepare_turn(turn)
            print("preparing data done")
            print("prepared dial", prepared_dial)
            # import time
            # time.sleep(10)
            dial_result = x.predict_turn(prepared_dial)
            print("dial result", dial_result)
            dev_bleu, dev_success, dev_match = x.evaluate_turn(dial_result)
            print("dev_bleu",dev_bleu)
            print("dev_success",dev_success) 
            print("dev_match",dev_match)  