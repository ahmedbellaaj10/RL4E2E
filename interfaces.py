from abc import ABC , abstractmethod
import argparse
import json
from multiprocessing import context
import os
import random
import yaml
import torch
import subprocess


import numpy as np
import torch

from galaxy.args import HParams, parse_args
from galaxy.args import str2bool
from galaxy.data.dataset import Dataset
from galaxy.data.field import BPETextField, MultiWOZBPETextField
from galaxy.models.model_base import ModelBase
from galaxy.models.generator import Generator
from galaxy.utils.eval import MultiWOZEvaluator
from galaxy.trainers.trainer import Trainer , MultiWOZTrainer

# ----------------------------------------
from transformers import T5Tokenizer
model_path = r'/home/ahmed/TOD_TEST/Models/pptod/checkpoints/base'
tokenizer = T5Tokenizer.from_pretrained(model_path)


from Models.pptod.E2E_TOD.modelling.T5Model import T5Gen_Model
from Models.pptod.E2E_TOD.ontology import sos_eos_tokens
special_tokens = sos_eos_tokens


class Interface(ABC):

    @abstractmethod
    def prepare_interface(self):
        pass

    @abstractmethod
    def predict_turn(self):
        pass

    @abstractmethod
    def predict_dialogue(self):
        pass

    @abstractmethod
    def evaluate_turn(self):
        pass

    @abstractmethod
    def evaluate_dialogue(self):
        pass

class GalaxyInterface(Interface):


    parser = argparse.ArgumentParser()

    # parser.add_argument("--do_train", type=str2bool, default=False,
    #                     help="Whether to run training on the train dataset.")

#     export CUDA_VISIBLE_DEVICES=0

# # Parameters.
# DATA_NAME=multiwoz
# PROJECT_NAME=GALAXY
# MODEL=UnifiedTransformer
# PROJECT_ROOT=/home/myself/${PROJECT_NAME}
# SAVE_ROOT=/data_hdd/myself/${PROJECT_NAME}
# VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
# VERSION=2.0
# LOAD_MODEL_DIR=110-35
# LOAD_MODEL_NAME=state_epoch_7
# INIT_CHECKPOINT=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/${LOAD_MODEL_DIR}/${LOAD_MODEL_NAME}
# WITH_JOINT_ACT=false
# USE_TRUE_PREV_BSPN=false
# USE_TRUE_PREV_ASPN=false
# USE_TRUE_PREV_RESP=false
# USE_TRUE_CURR_BSPN=false
# USE_TRUE_CURR_ASPN=false
# USE_TRUE_DB_POINTER=false
# USE_ALL_PREVIOUS_CONTEXT=true
# BATCH_SIZE=1
# BEAM_SIZE=1
# NUM_GPU=1
# SEED=10
# SAVE_DIR=${SAVE_ROOT}/outputs/${DATA_NAME}${VERSION}/${LOAD_MODEL_DIR}.infer

# # Main run.
# python -u run.py \
#   --do_infer=true \
#   --model=${MODEL} \
#   --save_dir=${SAVE_DIR} \
#   --data_name=${DATA_NAME} \
#   --data_root=${PROJECT_ROOT} \
#   --vocab_path=${VOCAB_PATH} \
#   --init_checkpoint=${INIT_CHECKPOINT} \
#   --with_joint_act=${WITH_JOINT_ACT} \
#   --use_true_prev_bspn=${USE_TRUE_PREV_BSPN} \
#   --use_true_prev_aspn=${USE_TRUE_PREV_ASPN} \
#   --use_true_prev_resp=${USE_TRUE_PREV_RESP} \
#   --use_true_curr_bspn=${USE_TRUE_CURR_BSPN} \
#   --use_true_curr_aspn=${USE_TRUE_CURR_ASPN} \
#   --use_true_db_pointer=${USE_TRUE_DB_POINTER} \
#   --use_all_previous_context=${USE_ALL_PREVIOUS_CONTEXT} \
#   --batch_size=${BATCH_SIZE} \
#   --beam_size=${BEAM_SIZE} \
#   --version=${VERSION} \
#   --gpu=${NUM_GPU} \
#   --seed=${SEED} \

    def __init__(self):
        # with open(r'/home/ahmed/TOD_TEST/Models/GALAXY/config.yaml') as file:
        #     doc = yaml.load(file, Loader=yaml.FullLoader)

        #     args = yaml.dump(doc, sort_keys=True)
        #     args = json.dumps(args, indent=2)
        #     print(args)
        # try :
        parser = argparse.ArgumentParser()
        stream = open("/home/ahmed/TOD_TEST/Models/GALAXY/config.yaml", 'r')
        args = yaml.load_all(stream, Loader=yaml.FullLoader)
        for doc in args:
            for key, value in doc.items():

                print(key + " : " + str(value))
                # if type(value) is list:
                #     print(str(len(value)))

                option = "--"+str(key)
                print("OPTION !!!!!!!!",option)
                require = False
                if key in ['vocab_path', 'data_name', 'save_dir']:
                    require = True
                if type(value).__name__ == int :
                    parser.add_argument(option , type=int , help=option , required= require , default=1 )  
                if type(value).__name__ == bool :
                    parser.add_argument(option , type=bool , help=option ,required= require ,default=True)  
                if type(value).__name__ == str :
                    parser.add_argument(option , type=str , help=option ,required= require, default="this is a string test")  
        # return
        # parser = argparse.ArgumentParser()

        # for keys , values in args.items():
        #     print(keys)
        # return

        # parser.add_argument("--do_train", type=str2bool, default=False,
        #                     help="Whether to run training on the train dataset.")
        # parser.add_argument("--do_test", type=str2bool, default=False,
        #                     help="Whether to run evaluation on the dev dataset.")
        # parser.add_argument("--do_infer", type=str2bool, default=False,
        #                     help="Whether to run inference on the test dataset.")
        # parser.add_argument("--num_infer_batches", type=int, default=None,
        #                     help="The number of batches need to infer.\n"
        #                         "Stay 'None': infer on entrie test dataset.")
        # parser.add_argument("--hparams_file", type=str, default=None,
        #                     help="Loading hparams setting from file(.json format).")
        BPETextField.add_cmdline_argument(parser)
        Dataset.add_cmdline_argument(parser)
        Trainer.add_cmdline_argument(parser)
        
        hparams = parse_args(parser)
        return
        print(json.dumps(hparams, indent=2))
        return
        ModelBase.add_cmdline_argument(parser)
        
        Generator.add_cmdline_argument(parser)
        # parser.parse_args()
        print("parser ", parser)
        hparams = parse_args(parser)
        hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 1
        print(json.dumps(hparams, indent=2))

        bpe = MultiWOZBPETextField(hparams)
        evaluator = MultiWOZEvaluator(reader=bpe)
        hparams.Model.num_token_embeddings = bpe.vocab_size
        hparams.Model.num_turn_embeddings = bpe.max_ctx_turn + 1
        generator = Generator.create(hparams, reader=bpe)

        # construct model
        model = ModelBase.create(hparams, generator=generator)
        print("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))

        # multi-gpu setting
        if hparams.gpu > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # construct trainer
        try: 
            trainer = MultiWOZTrainer(model, self.to_tensor, hparams, reader=bpe, evaluator=evaluator)
        except :
            raise NotImplementedError("Other dataset's trainer to be implemented !")

        # set optimizer and lr_scheduler
        # if hparams.do_train:
        #     trainer.set_optimizers()

        # load model, optimizer and lr_scheduler
        trainer.load()
        print("7chineh")
        # except :
        #     print("ta7cha")

    def prepare_interface(self):
        pass

    def to_tensor(self, array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if self.hparams.use_gpu else array

    
    def predict_turn(self , dialogue , turn_idx , pv_turn=None):
        with torch.no_grad():
            for fn, dial in dialogue.items(): 
                print("chnou l7all", {fn : dial})
                dialogue = self.reader._get_encoded_data(fn, dial)
            turn = dialogue[turn_idx]        
            first_turn = (turn_idx == 0)
            if first_turn :
                pv_turn = {}
            inputs, prompt_id = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
            batch, batch_size = self.reader.collate_fn_multi_turn(samples=[inputs])
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            if self.reader.use_true_curr_bspn:  # generate act, response
                max_len = 60
                if not self.reader.use_true_curr_aspn:
                    max_len = 80
                outputs = self.func_model.infer(inputs=batch, start_id=prompt_id,
                                                eos_id=self.reader.eos_r_id, max_gen_len=max_len)   
                
                # resp_gen, need to trim previous context
                generated = outputs[0].cpu().numpy().tolist()
                
                try:
                    decoded = self.decode_generated_act_resp(generated)
                except ValueError as exception:
                    self.logger.info(str(exception))
                    self.logger.info(self.tokenizer.decode(generated))
                    decoded = {'resp': [], 'bspn': [], 'aspn': []}
            else:  # predict bspn, access db, then generate act and resp
                outputs = self.func_model.infer(inputs=batch, start_id=prompt_id,
                                                eos_id=self.reader.eos_b_id, max_gen_len=60)

                generated_bs = outputs[0].cpu().numpy().tolist()
                bspn_gen = self.decode_generated_bspn(generated_bs)
                # check DB result
                if self.reader.use_true_db_pointer:
                    db = turn['db']
                else:
                    db_result = self.reader.bspan_to_DBpointer(self.tokenizer.decode(bspn_gen),
                                                                turn['turn_domain'])
                    assert len(turn['db']) == 4
                    book_result = turn['db'][2]
                    assert isinstance(db_result, str)
                    db = [self.reader.sos_db_id] + \
                            self.tokenizer.convert_tokens_to_ids([db_result]) + \
                            [book_result] + \
                            [self.reader.eos_db_id]
                    prompt_id = self.reader.sos_a_id

                prev_input = torch.tensor(bspn_gen + db)
                if self.func_model.use_gpu:
                    prev_input = prev_input.cuda()
                outputs_db = self.func_model.infer(inputs=batch, start_id=prompt_id,
                                                    eos_id=self.reader.eos_r_id, max_gen_len=80,
                                                    prev_input=prev_input)
                generated_ar = outputs_db[0].cpu().numpy().tolist()
                try:
                    decoded = self.decode_generated_act_resp(generated_ar)
                    decoded['bspn'] = bspn_gen
                except ValueError as exception:
                    self.logger.info(str(exception))
                    self.logger.info(self.tokenizer.decode(generated_ar))
                    decoded = {'resp': [], 'bspn': [], 'aspn': []}

            turn['resp_gen'] = decoded['resp']
            turn['bspn_gen'] = turn['bspn'] if self.reader.use_true_curr_bspn else decoded['bspn']
            turn['aspn_gen'] = turn['aspn'] if self.reader.use_true_curr_aspn else decoded['aspn']
            turn['dspn_gen'] = turn['dspn']

            pv_turn['labels'] = inputs['labels']  # all true previous context
            pv_turn['resp'] = turn['resp'] if self.reader.use_true_prev_resp else decoded['resp']
            if not self.reader.use_true_curr_bspn:
                pv_turn['bspn'] = turn['bspn'] if self.reader.use_true_prev_bspn else decoded['bspn']
                pv_turn['db'] = turn['db'] if self.reader.use_true_prev_bspn else db
            pv_turn['aspn'] = turn['aspn'] if self.reader.use_true_prev_aspn else decoded['aspn']
            tmp_dialog_result = self.reader.inverse_transpose_turn([turn])
            results, _ = self.reader.wrap_result_lm(tmp_dialog_result)
            bleu, success, match = self.evaluator.validation_metric([results[1]])
            print('results :', results)
            return bleu , results , tmp_dialog_result , pv_turn


    def predict_dialogue(self):
        pass

    def evaluate_turn(self):
        pass

    def evaluate_dialogue(self):
        pass

# def parse_args(parser, arguments):
#     """ Parse hyper-parameters from cmdline. """
#     parsed = parser.parse_args(arguments)
#     args = HParams()
#     optional_args = parser._action_groups[1]
#     for action in optional_args._group_actions[1:]:
#         arg_name = action.dest
#         args[arg_name] = getattr(parsed, arg_name)
#     for group in parser._action_groups[2:]:
#         group_args = HParams()
#         for action in group._group_actions:
#             arg_name = action.dest
#             group_args[arg_name] = getattr(parsed, arg_name)
#         if len(group_args) > 0:
#             args[group.title] = group_args
#     return args

class PptodInterface(Interface): 

    def __init__(self):
        self.model = T5Gen_Model(model_path, tokenizer, special_tokens, dropout=0.0, 
        add_special_decoder_token=True, is_training=False)
        self.model.eval()
        self.sos_context_token_id = tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
        self.sos_r_token_id, self.eos_r_token_id, self.sos_ic_token_id, self.eos_ic_token_id = \
        tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
        '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>', '<sos_d>', '<eos_d>'])
        # self.bs_prefix_text = 'translate dialogue to belief state:'
        # self.bs_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.bs_prefix_text))
        # self.da_prefix_text = 'translate dialogue to dialogue action:'
        # self.da_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.da_prefix_text))
        self.nlg_prefix_text = 'translate dialogue to system response:'
        self.nlg_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.nlg_prefix_text))
        # self.ic_prefix_text = 'translate dialogue to user intent:'
        # self.ic_prefix_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.ic_prefix_text))

    def prepare_context(self, dial):
        return context
            

    def predict_turn(self):
        dialogue_context = "<sos_u> can i reserve a five star place for thursday night at 3:30 for 2 people <eos_u> <sos_r> i'm happy to assist you! what city are you dining in? <eos_r> <sos_u> seattle please. <eos_u>"
        context_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dialogue_context))
        input_id = self.nlg_prefix_id + [self.sos_context_token_id] + context_id + [self.eos_context_token_id]
        input_id = torch.LongTensor(input_id).view(1, -1)
        x = self.model.model.generate(input_ids = input_id, decoder_start_token_id = self.sos_r_token_id,
                    pad_token_id = self.pad_token_id, eos_token_id = self.eos_r_token_id, max_length = 128)
        print(self.model.tokenized_decode(x[0]))
        return self.model.tokenized_decode(x[0])

    def prepare_interface(self):
        print("prepare_interface")


    def predict_dialogue(self):
        print("predict_dialogue")

    def evaluate_turn(self):
        print("evaluate_turn")

    def evaluate_dialogue(self):
        print("evaluate_dialogue")


if __name__ == "__main__":
    x = PptodInterface()
    x.predict_turn()

