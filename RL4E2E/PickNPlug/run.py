import os
import random
import json
import argparse
import json
import logging
import os
import random
import time
import numpy as np
from tqdm import tqdm
import sys
from RL4E2E.utils.constants import BASELINE_PATH
from wrappers.interfaces import GalaxyInterface, PptodInterface
from pnp import PickNPlug
import argparse

import pandas as pd

def run(args,logging_path):
    
    for position in ["before" , "after" , "wherever"]:
        for words in [1,3,5]:
            pnp = PickNPlug(position , words)
            log_path = os.path.join(logging_path,"pnp"+str(words)+position+"_output.log")
            if args.model == "galaxy":
                x = GalaxyInterface(args.version, log_path)
            else :
                x = PptodInterface(args.version, log_path)
            logging.basicConfig(level=logging.INFO, filename=log_path)
            infos  = []
            # try :
            if args.save_dir:
                save_dir = os.path.join(os.path.join(args.save_dir, args.model),args.version+"k"+str(words)+position)
                os.makedirs(save_dir, exist_ok=True)
            start_time = time.time()
            pbar = tqdm(range(args.iterations))
            seed_value = random.randrange(sys.maxsize)
            random.seed(seed_value)
            if args.model == "galaxy":
                try:
                    for a in pbar:
                        info = {}
                        idx , dial_title, dial = x.get_dialogue("test")
                        info['dial_title'] = dial_title
                        goals = x.get_dialogue_goal(dial)
                        info['goals']= goals
                        encoded = x.encode_dialogue(dial_title,dial)
                        results , bleu , success , match = x.predict_dialogue(encoded)
                        attacked_dial , validity , valids = pnp.attack_galaxy(dial_title,dial,goals)
                        info['validity'] = validity
                        info['valid']= valids
                        encoded_attacked = x.encode_dialogue(dial_title,attacked_dial)
                        results_ , bleu_ , success_ , match_ = x.predict_dialogue(encoded_attacked)
                        info['bleu_origin'] = bleu
                        info['success_origin'] = success
                        info['inform_origin'] = match
                        info['bleu_'] = bleu_
                        info['success_'] = success_
                        info['inform_'] = match_
                        infos.append(info)
                except KeyboardInterrupt:
                    infos.append(info)
                    file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
                    json.dump(infos, file, indent=4)   
                    file.close()   
                except :
                    infos.append(info)
                    file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
                    json.dump(infos, file, indent=4)   
                    file.close() 
            else :
                try :
                    data = x.data
                    for a in pbar:
                        info = {}
                        idx , dial_title, dial = x.get_dialogue("test")
                        info['dial_title'] = dial_title
                        print("got dialogue")
                        goals = x.get_dialogue_goal(dial_title, args.version)
                        info['goals']= goals
                        print("got dialogue goals")
                        attacked_dial , validity , valids = pnp.attack_pptod(dial,goals)
                        info['validity'] = str(validity)
                        info['valid']= valids
                        encoded , encoded_attacked = x.encode_dialogue(dial) , x.encode_dialogue(attacked_dial)
                        print("attacked got encoded")
                        result , attacked_result = x.predict_dial(encoded) , x.predict_dial(encoded_attacked)
                        print("attacked got predicted")
                        bleu , success , match = x.evaluate(result)
                        info['bleu_origin'] = bleu
                        info['success_origin'] = success
                        info['inform_origin'] = match
                        bleu_ , success_ , match_ = x.evaluate(attacked_result)
                        info['bleu_'] = bleu_
                        info['success_'] = success_
                        info['inform_'] = match_
                        infos.append(info)

                except KeyboardInterrupt:
                    infos.append(info)
                    file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
                    json.dump(infos, file, indent=4)   
                    file.close()   
                except :
                    infos.append(info)
                    file = open(os.path.join(save_dir,"infos_eval_backup.json"), "w")
                    json.dump(infos, file, indent=4)   
                    file.close()      
            end_time = time.time()
            file = open(os.path.join(save_dir,"infos_eval.json"), "w")
            json.dump(infos, file, indent=4)
            file.close()
            logging.info("Took %.2f seconds" % (end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=512, help='Random seed.', type=int)
    parser.add_argument('--iterations', default=100, help='Episodes over which to evaluate after training.', type=int) # episodes = 1000
    parser.add_argument('--save_dir', default=os.path.join(BASELINE_PATH,"results/"), help='Output directory.', type=str)
    parser.add_argument('--action', default="test", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    parser.add_argument('--version', default="2.0", choices=["2.0", "2.1"], help="the multiwoz version we want to use", type=str) 
    parser.add_argument('--num_selected_actions', default=1, help="how many actions to apply simultaniously", type=int) 
    args = parser.parse_args()
    mode = "test"
    logging_path = os.path.join(BASELINE_PATH,"results/pnp/"+args.model+"/"+args.version)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    
    parser.add_argument("--words", type= int, default=5 , choices=[1,3,5])
    parser.add_argument("--position", type= str, default="before" , choices=["before" , "after" , "wherever"])
    args = parser.parse_args()
    
    run(args,logging_path)
    