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



import numpy as np
import torch
import torch.nn.functional as F




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
        from galaxy.data.dataset import Dataset
        from galaxy.data.field import BPETextField, MultiWOZBPETextField, CamRestBPETextField, KvretBPETextField
        from galaxy.trainers.trainer import Trainer, MultiWOZTrainer, CamRestTrainer, KvretTrainer
        from galaxy.models.model_base import ModelBase
        from galaxy.models.generator import Generator
        from galaxy.utils.eval import MultiWOZEvaluator, CamRestEvaluator, KvretEvaluator
        from galaxy.args import parse_args
        
        parser = argparse.ArgumentParser(conflict_handler="resolve")
        stream = open("/home/ahmed/RL4E2E/Models/galaxy_config.yaml", 'r')
        args = yaml.load_all(stream, Loader=yaml.FullLoader)
        for doc in args:
            for key, value in doc.items():
                option = "--"+str(key)
                # if option in ['--vocab_path' , '--version', '--data_root', '--data_processed' , '--filtered', '--max_len' ,]:
                #     continue
                if type(value).__name__ == 'bool' :
                    parser.add_argument(option , type=bool , help=option ,default=value)  
                if type(value).__name__ == 'str' :
                    parser.add_argument(option , type=str , help=option , default=value)  
                if type(value).__name__ == 'int' :
                    parser.add_argument(option , type=int , help=option , default=value) 
                if type(value).__name__ == 'float' :
                    parser.add_argument(option , type=float , help=option , default=value) 
        BPETextField.add_cmdline_argument(parser)
        Dataset.add_cmdline_argument(parser)
        Trainer.add_cmdline_argument(parser)
        
        ModelBase.add_cmdline_argument(parser)
        
        Generator.add_cmdline_argument(parser)
        hparams = parse_args(parser)
        hparams.use_gpu = torch.cuda.is_available() and hparams.gpu >= 1
        print(json.dumps(hparams, indent=2))
        sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
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


class PptodInterface(Interface): 

    def __init__(self):
        
        if torch.cuda.is_available():
            print ('Cuda is available.')
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            if torch.cuda.device_count() > 1:
                multi_gpu_training = True
                print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
            else:
                print ('Using single GPU training.')
        else:
            pass
    
        args = parse_config()
        print("args" , args)
        device = torch.device('cuda')
        sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")
        from dataclass import MultiWozData
        from config import Config
        from eval import MultiWozEvaluator
        cfg = Config(args.data_path_prefix)
        assert args.model_name.startswith('t5')
        from transformers import T5Tokenizer

        if args.pretrained_path != 'None':
            ckpt_name = get_checkpoint_name(args.pretrained_path)
            pretrained_path = args.pretrained_path + '/' + ckpt_name

        if args.pretrained_path != 'None':
            print ('Loading Pretrained Tokenizer...')
            tokenizer = T5Tokenizer.from_pretrained(pretrained_path)
        else:
            tokenizer = T5Tokenizer.from_pretrained(args.model_name)

        if args.use_db_as_input == 'True':
            use_db_as_input = True
        elif args.use_db_as_input == 'False':
            use_db_as_input = False
        else:
            raise Exception('Wrong Use DB Mode!!!')

        if args.add_prefix == 'True':
            add_prefix = True
        elif args.add_prefix == 'False':
            add_prefix = False
        else:
            raise Exception('Wrong Prefix Mode!!!')

        if args.add_special_decoder_token == 'True':
            add_special_decoder_token = True
        elif args.add_special_decoder_token == 'False':
            add_special_decoder_token = False
        else:
            raise Exception('Wrong Add Special Token Mode!!!')

        data = MultiWozData(args.model_name, tokenizer, cfg, args.data_path_prefix, shuffle_mode=args.shuffle_mode, 
            data_mode='train', use_db_as_input=use_db_as_input, add_special_decoder_token=add_special_decoder_token, 
            train_data_ratio=0.01)

        print ('Data loaded')
        evaluator = MultiWozEvaluator(data.reader, cfg)

        print ('Start loading model...')
        assert args.model_name.startswith('t5')
        from Models.pptod.E2E_TOD.modelling.T5Model import T5Gen_Model
        if args.pretrained_path != 'None':
            model = T5Gen_Model(pretrained_path, data.tokenizer, data.special_token_list, dropout=0.0, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)
        else:
            model = T5Gen_Model(args.model_name, data.tokenizer, data.special_token_list, dropout=0.0, 
                add_special_decoder_token=add_special_decoder_token, is_training=True)

        if cuda_available:
            model = model.to(device)
        else:
            pass
        model.eval()
        print ('Model loaded')

    def prepare_context(self, dial):
        return context
            

    def predict_turn(self):
       print("predict_turn")

    def prepare_interface(self):
        print("prepare_interface")


    def predict_dialogue(self):
        print("predict_dialogue")

    def evaluate_turn(self):
        print("evaluate_turn")

    def evaluate_dialogue(self):
        print("evaluate_dialogue")


if __name__ == "__main__":
    x = GalaxyInterface()



