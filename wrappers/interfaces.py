from abc import ABC , abstractmethod
import argparse
import json
from multiprocessing import context
import os
import random
import yaml
import torch
import subprocess
import sys
import pandas as pd



import numpy as np
import torch
import torch.nn.functional as F


from adapters import MyMultiWozData
from adapters import *


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
    
    def __init__(self):
        sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
        from pprint import pprint

        # pprint(sys.path)
        from galaxy.data.dataset import Dataset
        from galaxy.data.field import MultiWOZBPETextField
        from galaxy.trainers.trainer import  MultiWOZTrainer 
        from galaxy.models.model_base import ModelBase
        from galaxy.models.generator import Generator
        from galaxy.utils.eval import MultiWOZEvaluator
        from galaxy.args import parse_args
        from galaxy.args import str2bool
        from adapters import Hparams , BPETextField , Data , Trainer , Model , Gene

        # try :
        #     from config import Config
        # except :
        #     print('no')
        # parser = argparse.ArgumentParser()
        # #conflict_handler="resolve")
        stream = open("/home/ahmed/RL4E2E/Models/galaxy_config.yaml", 'r')
        # sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
        
        # stream2 = open("bus_db.json", 'r')
        args = yaml.load_all(stream, Loader=yaml.FullLoader)
        for doc in args:
            textfield = BPETextField(vocab_path=doc["vocab_path"], version=doc["version"],data_root=doc["data_root"],data_processed=doc["data_processed"], filtered=doc["filtered"], max_len=doc["max_len"], min_utt_len=doc["min_utt_len"], max_utt_len=doc["max_utt_len"], min_ctx_turn=doc["min_ctx_turn"], max_ctx_turn=doc["max_ctx_turn"], tokenizer_type=doc["tokenizer_type"])
            data = Data(data_dir=doc["data_dir"],data_name=doc["data_name"])
            trainer = Trainer(seed=doc["seed"], gpu=doc["gpu"], valid_metric_name=doc["valid_metric_name"],num_epochs=doc["num_epochs"], save_dir=doc["save_dir"], token_loss=doc["token_loss"], batch_size=doc["batch_size"] , log_steps=doc["log_steps"], valid_steps=doc["valid_steps"], save_checkpoint=doc["save_checkpoint"], shuffle=doc["shuffle"], sort_pool_size=doc["sort_pool_size"])
            model_ = Model(init_checkpoint=doc["init_checkpoint"] , model=doc["model"], num_token_embeddings=doc["num_token_embeddings"], num_pos_embeddings=doc["num_pos_embeddings"], num_type_embeddings=doc["num_type_embeddings"], num_turn_embeddings=doc["num_turn_embeddings"], num_act=doc["num_act"], num_heads=doc["num_heads"], num_layers=doc["num_layers"] , hidden_dim=doc["hidden_dim"], padding_idx=doc["padding_idx"], dropout=doc["dropout"], embed_dropout=doc["embed_dropout"], attn_dropout=doc["attn_dropout"], ff_dropout=doc["ff_dropout"], use_discriminator=doc["use_discriminator"], dis_ratio=doc["dis_ratio"], bce_ratio=doc["bce_ratio"],pos_trainable= doc["pos_trainable"], with_joint_act=doc["with_joint_act"], with_rdrop_act=doc["with_rdrop_act"], initializer_range=doc["initializer_range"], lr=doc["lr"], weight_decay=doc["weight_decay"], gradient_accumulation_steps=doc["gradient_accumulation_steps"], warmup_steps=doc["warmup_steps"], max_grad_norm=doc["max_grad_norm"])
            gene = Gene(generator=doc["generator"],min_gen_len=doc["min_gen_len"], max_gen_len=doc["max_gen_len"], use_true_prev_bspn= doc["use_true_prev_bspn"] , use_true_prev_aspn= doc["use_true_prev_aspn"],use_true_db_pointer=doc["use_true_db_pointer"], use_true_prev_resp= doc["use_true_prev_resp"], use_true_curr_bspn=doc["use_true_curr_bspn"], use_true_curr_aspn=doc["use_true_curr_aspn"], use_all_previous_context=doc["use_all_previous_context"], use_true_bspn_for_ctr_eval=doc["use_true_bspn_for_ctr_eval"], use_true_domain_for_ctr_eval=doc["use_true_domain_for_ctr_eval"], beam_size=doc["beam_size"], length_average=doc["length_average"], length_penalty=doc["length_penalty"], ignore_unk=doc["ignore_unk"])
            self.hparams = Hparams(do_train=doc["do_train"], do_test=doc["do_test"], do_infer=doc["do_infer"], num_infer_batches=doc["num_infer_batches"],hparams_file=doc["hparams_file"], vocab_path=doc["vocab_path"], version=doc["version"],data_root=doc["data_root"],data_processed=doc["data_processed"], filtered=doc["filtered"], max_len=doc["max_len"], min_utt_len=doc["min_utt_len"], max_utt_len=doc["max_utt_len"], min_ctx_turn=doc["min_ctx_turn"], max_ctx_turn=doc["max_ctx_turn"], tokenizer_type=doc["tokenizer_type"], data_dir=doc["data_dir"],data_name=doc["data_name"], seed=doc["seed"], gpu=doc["gpu"], valid_metric_name=doc["valid_metric_name"],num_epochs=doc["num_epochs"], save_dir=doc["save_dir"], token_loss=doc["token_loss"], batch_size=doc["batch_size"] , log_steps=doc["log_steps"], valid_steps=doc["valid_steps"], save_checkpoint=doc["save_checkpoint"], shuffle=doc["shuffle"], sort_pool_size=doc["sort_pool_size"], init_checkpoint=doc["init_checkpoint"] , model=doc["model"],
            num_token_embeddings=doc["num_token_embeddings"], num_pos_embeddings=doc["num_pos_embeddings"], num_type_embeddings=doc["num_type_embeddings"], num_turn_embeddings=doc["num_turn_embeddings"], num_act=doc["num_act"], num_heads=doc["num_heads"], num_layers=doc["num_layers"] , hidden_dim=doc["hidden_dim"], padding_idx=doc["padding_idx"], dropout=doc["dropout"], embed_dropout=doc["embed_dropout"], attn_dropout=doc["attn_dropout"], ff_dropout=doc["ff_dropout"], use_discriminator=doc["use_discriminator"], dis_ratio=doc["dis_ratio"], bce_ratio=doc["bce_ratio"],pos_trainable= doc["pos_trainable"], with_joint_act=doc["with_joint_act"], with_rdrop_act=doc["with_rdrop_act"], initializer_range=doc["initializer_range"], lr=doc["lr"], weight_decay=doc["weight_decay"], gradient_accumulation_steps=doc["gradient_accumulation_steps"], warmup_steps=doc["warmup_steps"], max_grad_norm=doc["max_grad_norm"], generator=doc["generator"],min_gen_len=doc["min_gen_len"], max_gen_len=doc["max_gen_len"], use_true_prev_bspn= doc["use_true_prev_bspn"] , use_true_prev_aspn= doc["use_true_prev_aspn"],use_true_db_pointer=doc["use_true_db_pointer"], use_true_prev_resp= doc["use_true_prev_resp"], use_true_curr_bspn=doc["use_true_curr_bspn"], use_true_curr_aspn=doc["use_true_curr_aspn"], use_all_previous_context=doc["use_all_previous_context"], use_true_bspn_for_ctr_eval=doc["use_true_bspn_for_ctr_eval"], use_true_domain_for_ctr_eval=doc["use_true_domain_for_ctr_eval"], beam_size=doc["beam_size"], length_average=doc["length_average"], length_penalty=doc["length_penalty"], ignore_unk=doc["ignore_unk"], use_gpu=doc["use_gpu"], BPETextField=textfield, Dataset=data, Trainer=trainer, Generator= gene , Model=model_)
        
        self.hparams.use_gpu = torch.cuda.is_available() and self.hparams.gpu >= 1
        self.data = json.load(open("/home/ahmed/RL4E2E/Models/GALAXY/data/multiwoz2.0/data_for_galaxy.json"))
        # print(json.dumps(hparams, indent=2))
        os.chdir('/home/ahmed/RL4E2E/Models/GALAXY/')
        self.reader = MultiWOZBPETextField(self.hparams)
        self.evaluator = MultiWOZEvaluator(reader=self.reader)
        self.hparams.Model.num_token_embeddings = self.reader.vocab_size
        self.hparams.num_token_embeddings = self.reader.vocab_size
        print("reader.vocab_size", self.reader.vocab_size)
        print("reader.hparams.Model.num_token_embeddings", self.hparams.Model.num_token_embeddings)
        # exit()
        self.hparams.Model.num_turn_embeddings = self.reader.max_ctx_turn + 1
        generator = Generator.create(self.hparams, reader=self.reader)
        print(self.hparams)
        # construct model
        model = ModelBase.create(self.hparams, generator=generator)
        print("Total number of parameters in networks is {}".format(sum(x.numel() for x in model.parameters())))

        # multi-gpu setting
        if self.hparams.gpu > 1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # construct trainer
        try: 
            self.trainer = MultiWOZTrainer(model, self.to_tensor, self.hparams, reader=self.reader, evaluator=self.evaluator)
        except :
            raise NotImplementedError("Other dataset's trainer to be implemented !")

        # set optimizer and lr_scheduler
        # if hparams.do_train:
        #     trainer.set_optimizers()

        # load model, optimizer and lr_scheduler
        self.trainer.load()

    def prepare_interface(self):
        pass

    def to_tensor(self, array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if self.hparams.use_gpu else array

    def get_dialogue(self, dial_name):
        return self.data[dial_name]

    def get_turn(self , dial, turn_num):
        dial_copy = copy.deepcopy(dial)
        dial_copy["log"] = dial_copy["log"][turn_num:turn_num+1]
        return dial_copy

    def copy_dial_or_turn(self , dial_turn):
        return copy.deepcopy(dial_turn)

    def get_turn_with_context(self , dial, turn_num):
        dial_copy = copy.deepcopy(dial)
        dial_copy["log"] = dial_copy["log"][:turn_num+1]
        return dial_copy

    def get_utterance(self, dial , turn_num):
        return dial["log"][turn_num]["user_delex"]

    def set_utterance(self, dial, turn_num, transformed_sentence):
        dial["log"][turn_num]["user_delex"] = transformed_sentence

    def encode(self , dial_name, turn_or_dial):
        encoded = self.reader._get_encoded_data(dial_name, turn_or_dial)
        # print("encoded", encoded)
        return encoded

    def predict_turn(self , turn_with_context , idx, pv_turn=None):
        with torch.no_grad():
            
            # for dial_idx, dialog in enumerate(turn_with_context):
            #     print("dialog", dialog)
            pv_turn = {}
            for turn_idx, turn in enumerate(turn_with_context):
                    # print(turn)
                    first_turn = (turn_idx == 0)
                    inputs, prompt_id = self.reader.convert_turn_eval(turn, pv_turn, first_turn)
                    batch, batch_size = self.reader.collate_fn_multi_turn(samples=[inputs])
                    batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
                    if self.reader.use_true_curr_bspn:  # generate act, response
                        max_len = 60
                        if not self.reader.use_true_curr_aspn:
                            max_len = 80
                        outputs = self.trainer.func_model.infer(inputs=batch, start_id=prompt_id,
                                                        eos_id=self.reader.eos_r_id, max_gen_len=max_len)
                        # resp_gen, need to trim previous context
                        generated = outputs[0].cpu().numpy().tolist()
                        try:
                            decoded = self.trainer.decode_generated_act_resp(generated)
                        except ValueError as exception:
                            self.logger.info(str(exception))
                            self.logger.info(self.trainer.tokenizer.decode(generated))
                            decoded = {'resp': [], 'bspn': [], 'aspn': []}
                    else:  # predict bspn, access db, then generate act and resp
                        outputs = self.trainer.func_model.infer(inputs=batch, start_id=prompt_id,
                                                        eos_id=self.reader.eos_b_id, max_gen_len=60)
                        generated_bs = outputs[0].cpu().numpy().tolist()
                        bspn_gen = self.trainer.decode_generated_bspn(generated_bs)
                        # check DB result
                        if self.reader.use_true_db_pointer:
                            db = turn['db']
                        else:
                            db_result = self.reader.bspan_to_DBpointer(self.trainer.tokenizer.decode(bspn_gen),
                                                                       turn['turn_domain'])
                            assert len(turn['db']) == 4
                            book_result = turn['db'][2]
                            assert isinstance(db_result, str)
                            db = [self.reader.sos_db_id] + \
                                 self.trainer.tokenizer.convert_tokens_to_ids([db_result]) + \
                                 [book_result] + \
                                 [self.reader.eos_db_id]
                            prompt_id = self.reader.sos_a_id

                        prev_input = torch.tensor(bspn_gen + db)
                        if self.trainer.func_model.use_gpu:
                            prev_input = prev_input.cuda()
                        outputs_db = self.trainer.func_model.infer(inputs=batch, start_id=prompt_id,
                                                           eos_id=self.reader.eos_r_id, max_gen_len=80,
                                                           prev_input=prev_input)
                        generated_ar = outputs_db[0].cpu().numpy().tolist()
                        try:
                            decoded = self.trainer.decode_generated_act_resp(generated_ar)
                            decoded['bspn'] = bspn_gen
                        except ValueError as exception:
                            self.logger.info(str(exception))
                            self.logger.info(self.trainer.tokenizer.decode(generated_ar))
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
            # print("turn_with_context", turn_with_context)
            tmp_dialog_result = self.reader.inverse_transpose_turn(turn_with_context[::-1])
            results, _ = self.reader.wrap_result_lm(tmp_dialog_result)
            turn_output = [results[1]]
            bleu, success, match = self.evaluator.validation_metric(turn_output)
            # print('results :', results)
            return turn_output, bleu , tmp_dialog_result , pv_turn


class PptodInterface(Interface): 

    def __init__(self):
        
        if torch.cuda.is_available():
            print ('Cuda is available.')
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            if torch.cuda.device_count() > 1:
                multi_gpu_training = True
                print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
            else:
                print ('Using single GPU training.')
        else:
            pass
    
        self.args = parse_config()
        print("args" , self.args)
        device = torch.device('cuda')
        sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")
        from config import Config
        from eval import MultiWozEvaluator
        cfg = Config(self.args.data_path_prefix)
        assert self.args.model_name.startswith('t5')
        from transformers import T5Tokenizer

        if self.args.pretrained_path != 'None':
            ckpt_name = get_checkpoint_name(self.args.pretrained_path)
            pretrained_path = self.args.pretrained_path + '/' + ckpt_name

        if self.args.pretrained_path != 'None':
            print ('Loading Pretrained Tokenizer...')
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_name)

        if self.args.use_db_as_input == 'True':
            self.use_db_as_input = True
        elif self.args.use_db_as_input == 'False':
            self.use_db_as_input = False
        else:
            raise Exception('Wrong Use DB Mode!!!')

        if self.args.add_prefix == 'True':
            self.add_prefix = True
        elif self.args.add_prefix == 'False':
            self.add_prefix = False
        else:
            raise Exception('Wrong Prefix Mode!!!')

        if self.args.add_special_decoder_token == 'True':
            self.add_special_decoder_token = True
        elif self.args.add_special_decoder_token == 'False':
            self.add_special_decoder_token = False
        else:
            raise Exception('Wrong Add Special Token Mode!!!')

        self.data = MyMultiWozData(self.args.model_name, self.tokenizer, cfg, self.args.data_path_prefix, shuffle_mode=self.args.shuffle_mode, 
            data_mode='train', use_db_as_input=self.use_db_as_input, add_special_decoder_token=self.add_special_decoder_token, 
            train_data_ratio=0.01)

        print ('Data loaded')
        evaluator = MultiWozEvaluator(self.data.reader, cfg)



        print ('Start loading model...')
        assert self.args.model_name.startswith('t5')
        from Models.pptod.E2E_TOD.modelling.T5Model import T5Gen_Model
        if self.args.pretrained_path != 'None':
            self.model = T5Gen_Model(pretrained_path, self.data.tokenizer, self.data.special_token_list, dropout=0.0, 
                add_special_decoder_token=self.add_special_decoder_token, is_training=True)
        else:
            self.model = T5Gen_Model(self.args.model_name, self.data.tokenizer, self.data.special_token_list, dropout=0.0, 
                add_special_decoder_token=self.add_special_decoder_token, is_training=True)

        if self.cuda_available:
            self.model = self.model.to(device)
        else:
            pass
        self.model.eval()
        print ('Model loaded')

    def get_dialogue(self , dial_id , mode = 'dev'):
        return self.data.get_dialogue(dial_id)

    def copy_dialogue(self , dial):
        return copy.deepcopy(dial)

    def extract_turn(self , dial , num_turn):
        return dial[num_turn]

    def get_turn_with_context(self , dial , num_turn):
        return dial[:num_turn+1]

    def prepare_dialogue(self , dial):
        return self.data.prepare_dialogue(dial)

    def get_utterance(self , turn):
        return turn["usdx"]

    def set_utterance(self , turn , sentence):
        turn["usdx"] = sentence
        return

    
    # def get_encoded(self , turn):
    #     from Models.pptod.E2E_TOD.e2e_inference_utlis import e2e_batch_generate
    #     with torch.no_grad():
    #         ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
    #         self.input_contain_db=self.use_db_as_input
    #         eva_batch_size =self.args.number_of_gpu * self.args.batch_size_per_gpu
    #         one_inference_turn = self.data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, self.input_contain_db, 
    #             eva_batch_size=self.args.number_of_gpu * self.args.batch_size_per_gpu, turn=turn)
    #         print("one inference turn ",one_inference_turn)
    #         return one_inference_turn

    def get_encoded(self, turn):
        from Models.pptod.E2E_TOD.e2e_inference_utlis import e2e_batch_generate
        with torch.no_grad():
            ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
            input_contain_db=self.use_db_as_input
            eva_batch_size=self.args.number_of_gpu * self.args.batch_size_per_gpu
            dev_batch_list = self.data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, 
                eva_batch_size=self.args.number_of_gpu * self.args.batch_size_per_gpu , turn=turn)
            dev_batch_num_per_epoch = len(dev_batch_list)
            print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
        return dev_batch_list
    # def predict_turn(turn):
    #     with torch.no_grad():
    #         dial_result = []
    #         all_dev_result = []
    #         successful_dials = []
    #         for p_dev_idx in range(dev_batch_num_per_epoch):
    #             p.update(p_dev_idx)
    #             one_inference_batch = dev_batch_list[p_dev_idx]
    #             dev_batch_parse_dict = e2e_batch_generate(self.model, one_inference_batch, input_contain_db, self.data)
    #             print("one inference :" , one_inference_batch)
    #             # new to be deleted
    #             if dev_batch_parse_dict[0]['turn_num']==0:
    #                 dial_bleu, dial_success, dial_match = self.evaluator.validation_metric(dial_result)
    #                 import math
    #                 print ('The evaluation results are: Inform: {}, Success: {}, BLEU: {}'.format(dial_match, 
    #                     dial_success, dial_bleu))
    #                 if math.ceil(dial_success)==100 and math.ceil(dial_match)==100:
    #                     print("successful")
    #                     print(dial_result[0]['dial_id'])
    #                     successful_dials.append(dial_result[0]['dial_id']) 
    #                 dial_result = []
    #             for item in dev_batch_parse_dict:
    #                 dial_result.append(item)
    #             #end
    #             for item in dev_batch_parse_dict:
    #                 all_dev_result.append(item)
    #         p.finish()
            

    def predict_turn(self , one_inference_turn):
        print("one inference turn", one_inference_turn)
        from Models.pptod.E2E_TOD.e2e_inference_utlis import e2e_batch_generate
        x =  e2e_batch_generate(self.model, one_inference_turn, self.use_db_as_input, self.data)
        print("x ", x)
        return x

    def evaluate_turn(self , output ):
        dial_bleu , _ , _ = self.evaluator.validation_metric(output)
        return dial_bleu

    def prepare_interface(self):
        print("prepare_interface")


    def predict_dialogue(self):
        print("predict_dialogue")

    def evaluate_turn(self):
        print("evaluate_turn")

    def evaluate_dialogue(self):
        print("evaluate_dialogue")


if __name__ == "__main__":
    interface = "galaxy"
    # interface = "pptod"
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
        data = MyMultiWozData()
        x = PptodInterface()
        prepared_turn = x.prepare_dialogue(turn_with_context)
        results = x.predict_turn(encoded, 1)
        bleu = x.evaluate_turn(predicted_trun)
        print("bleu for this dialogue is : " , bleu)