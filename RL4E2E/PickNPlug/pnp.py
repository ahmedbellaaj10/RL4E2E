from fractions import Fraction
import logging
import os
import random
import json
import copy
from wrappers.interfaces import GalaxyInterface, PptodInterface
from RL4E2E.utils.constants import BASELINE_PATH

# MultiWOZBPETextField
class PickNPlug(object):

    def __new__(cls,*args, **kwargs):
         return super().__new__(cls)

    def __init__(self,adv_insert_mode="after",number_words=1):
        f = open(os.path.join(BASELINE_PATH,"src_data.json"),"r")        
        self.src = json.load(f)
        self.adv_insert_mode = adv_insert_mode
        """
        insert mode is 1 if we insert at the end for all the turns of the dialogue
                       2 if we insert at the start for all the turns of the dialogue
                       0 if we insert at the end or the start randomly at each turn of the dialogue
        """
        self.number_words = number_words
        """if 1 it's a sentence, else it's an n-sentence block"""

    # def transform_dial(self , dial , model):
    #     pass
        
    def attack_galaxy(self,dial_title,dial,goals):
        dialogue = copy.deepcopy(dial)
        name = dial_title
        turns = len(list(dialogue.values())[1])
        all_goals = list(set(self.src.keys()))
        # print(dialogue)
        for goal in goals:
            all_goals.remove(goal)
        validity = []
        for turn in range(turns):
            selected_goal = random.choice(all_goals)
            # while not choice:
            candidate = random.choice(self.src[selected_goal])
            logging.info(f"before {dialogue['log'][turn]['user']}")
            if len(list(candidate.keys())[0]) > self.number_words :
                inject= ' '.join(list(candidate.keys())[0].split()[:self.number_words])
                inject_delex = ' '.join(list(candidate.values())[0].split()[:self.number_words])
                dialogue['log'][turn]['user'] = self.plug_galaxy(dialogue['log'][turn]['user'],inject)
                dialogue['log'][turn]['user_delex'] = self.plug_galaxy(dialogue['log'][turn]['user_delex'],inject_delex)
                

            logging.info(f"after {dialogue['log'][turn]['user']}")
            validity.append(self.number_words / len(dialogue['log'][turn]['user'].split())  )
        valids = [valid< 0.25 for valid in validity]
        return dialogue , validity , valids

    def attack_pptod(self,dial,goals):
        dialogue = copy.deepcopy(dial)
        turns = len(dialogue)
        validity = []
        for turn in range(turns):

            all_goals = list(set(self.src.keys()))
            for goal in goals:
                all_goals.remove(goal)
            selected_goal = random.choice(all_goals)
            candidate = random.choice(self.src[selected_goal])
            print("candidate",candidate)
            logging.info(f"before {dialogue[turn]['user']}")
            if len(list(candidate.keys())[0]) > self.number_words :
                # try :
                inject= ' '.join(list(candidate.keys())[0].split()[::self.number_words])
                print("inject",inject)
                inject_delex = ' '.join(list(candidate.values())[0].split()[::self.number_words])
                dialogue[turn]['user'] = self.plug_pptod(dialogue[turn]['user'],inject)
                dialogue[turn]['usdx'] = self.plug_pptod(dialogue[turn]['usdx'],inject_delex)
                # except:
                #     pass
            logging.info(f"after {dialogue[turn]['user']}")
            validity.append(self.number_words / len(dialogue[turn]['user'].split()))
        valids = [valid< 0.25 for valid in validity]
        return dialogue , validity , valids

        # all_goals = list(set(self.src.keys()))
        # print(dialogue)
        # for goal in goals:
        #     all_goals.remove(goal)
        # for turn in range(turns):
        #     choice = False
        #     selected_goal = random.choice(all_goals)
        #     while not choice:
        #         candidate = random.choice(self.src[selected_goal])
        #         if len(list(candidate.keys())[0]) > self.number_words :
        #             try :
        #                 choice = True
        #                 inject= ' '.join(list(candidate.keys())[0].split()[:self.number_words])
        #                 inject_delex = ' '.join(list(candidate.values())[0].split()[:self.number_words])
        #                 dialogue[name]['log'][turn]['user'] = self.plug_galaxy(dialogue[name]['log'][turn]['user'],inject)
        #                 dialogue[name]['log'][turn]['user_delex'] = self.plug_galaxy(dialogue[name]['log'][turn]['user_delex'],inject_delex)
        #             except:
        #                 choice = False
        # return dialogue


    def plug_galaxy(self, utterance , source):
        if self.adv_insert_mode == "after":
            utterance = utterance+" "+source
        elif self.adv_insert_mode == "before":
            utterance = source+" "+utterance
        else :
            utterance = random.choice([utterance+source , source+utterance])
        return utterance

    def plug_pptod(self, utterance , source):
        begin = utterance.split()[0]
        end = utterance.split()[-1]
        utterance = ' '.join(utterance.split()[1:-1])
        if self.adv_insert_mode == "after":
            utterance = begin+" "+utterance+" "+source+" "+end
        elif self.adv_insert_mode == "before":
            utterance = begin+" "+source+" "+utterance+" "+end
        else :
            utterance = random.choice([begin+" "+utterance+" "+source+" "+end , begin+" "+source+" "+utterance+" "+end])
        return utterance
