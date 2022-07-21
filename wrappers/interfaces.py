from abc import ABC , abstractmethod
import json
import os
import random
import yaml
import sys
import copy
import torch
import torch.nn.functional as F

sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
from galaxy.trainers.trainer import  MultiWOZTrainer 
from galaxy.models.model_base import ModelBase
from galaxy.models.generator import Generator
from galaxy.utils.eval import MultiWOZEvaluator



sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")
from config import Config
from eval import MultiWozEvaluator
from transformers import T5Tokenizer
from Models.pptod.E2E_TOD.modelling.T5Model import T5Gen_Model
from Models.pptod.E2E_TOD.e2e_inference_utlis import e2e_batch_generate

from wrappers.adapters import Hparams , BPETextField , Data , Trainer , Model , Gene , MyMultiWOZBPETextField, MyMultiWozData

class Interface(ABC):

    @abstractmethod
    def predict_turn(self):
        pass

    @abstractmethod
    def evaluate_turn(self):
        pass

    @abstractmethod
    def choose_dialogue(self):
        pass

class GalaxyInterface(Interface):
    
    def __init__(self):
        sys.path.append("/home/ahmed/RL4E2E/Models/GALAXY")
        stream = open("/home/ahmed/RL4E2E/Models/galaxy_config.yaml", 'r')
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
        os.chdir('/home/ahmed/RL4E2E/Models/GALAXY/')
        self.reader = MyMultiWOZBPETextField(self.hparams)
        self.evaluator = MultiWOZEvaluator(reader=self.reader)
        self.hparams.Model.num_token_embeddings = self.reader.vocab_size
        self.hparams.num_token_embeddings = self.reader.vocab_size
        self.dev_data = self.reader.get_eval_data()
        self.test_data = self.reader.get_eval_data('test')
        print("test and dev data loaded")
        self.hparams.Model.num_turn_embeddings = self.reader.max_ctx_turn + 1
        generator = Generator.create(self.hparams, reader=self.reader)
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

        self.trainer.load()

    def to_tensor(self, array):
        """
        numpy array -> tensor
        """
        array = torch.tensor(array)
        return array.cuda() if self.hparams.use_gpu else array

    def get_dialogue(self, mode):
        if mode == 'dev':
            data = self.dev_data
        else :
            data = self.test_data
        idx = random.choice(len(data))
        dial_title = data[idx].keys()[0]
        return idx , dial_title, data[idx]

    def get_dialogue_length(dial):
        return len(dial.keys()[0]['log'])

    def get_turn(self , dial,  turn_num):
        dial_title = dial.keys()[0]
        dial_copy = copy.deepcopy(dial)
        dial_copy[dial_title]["log"] = dial_copy["log"][turn_num:turn_num+1]
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
            
            # print('results :', results)
            return turn_output , tmp_dialog_result , pv_turn

    def evaluate_turn(self , turn_output):
        bleu, success, match = self.evaluator.validation_metric(turn_output)
        return bleu, success, match


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
        device = torch.device('cpu')
        sys.path.append("/home/ahmed/RL4E2E/Models/pptod/E2E_TOD")
        # from config import Config
        # from eval import MultiWozEvaluator
        # from transformers import T5Tokenizer
        cfg = Config(self.args.data_path_prefix)
        assert self.args.model_name.startswith('t5')
        

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

        self.dev_data = self.data.dev_raw_data
        self.test_data = self.data.test_raw_data

        print ('Data loaded')
        self.evaluator = MultiWozEvaluator(self.data.reader, cfg)



        print ('Start loading model...')
        assert self.args.model_name.startswith('t5')
        

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

    def get_dialogue(self, mode = 'dev'):
        if mode == 'dev':
            data = self.dev_data
        else :
            data = self.test_data
        idx = random.choice(len(data))
        dial_title = data[idx][0]['dial_id']
        return idx , dial_title , data[idx]

    def get_dialogue_length(dial):
        return len(dial)

    def copy_dialogue(self , dial):
        return copy.deepcopy(dial)

    def get_turn(self , dial , num_turn):
        return [dial[num_turn]]

    def get_turn_with_context(self , dial , num_turn):
        return dial[:num_turn+1]

    def prepare_turn(self , dial):
        return self.data.prepare_dialogue(dial)[-1]

    def get_utterance(self , turn):
        return turn["usdx"]

    def set_utterance(self , turn , sentence):
        turn["usdx"] = sentence
        return

            

    def predict_turn(self , one_inference_turn):
        
        with torch.no_grad():
            turn = one_inference_turn
            import time
            time.sleep(10)
            ref_bs, ref_act, ref_db = False, False, False # we only consider e2e evaluation
            input_contain_db=self.use_db_as_input
            eva_batch_size=self.args.number_of_gpu * self.args.batch_size_per_gpu
            dev_batch_list = self.data.build_all_evaluation_batch_list(ref_bs, ref_act, ref_db, input_contain_db, eva_batch_size , turn )
            dev_batch_num_per_epoch = len(dev_batch_list)
            print ('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
            dial_result = []
            # for p_dev_idx in range(dev_batch_num_per_epoch):
            one_inference_batch = self.prepare_one_inference_batch(dev_batch_list)
            print("one_inference_batch",one_inference_batch)
            dev_batch_parse_dict = e2e_batch_generate(self.model, one_inference_batch, input_contain_db, self.data)
                
            for item in dev_batch_parse_dict:
                dial_result.append(item)
            return dial_result

    def prepare_one_inference_batch(self , one_item_batch_list):
        elements = one_item_batch_list[0]
        formatted = []
        for element in elements :
            formatted.append([element])
        return formatted


    def evaluate_turn(self , result):
        dev_bleu, dev_success, dev_match = self.evaluator.validation_metric(result)
        return dev_bleu, dev_success, dev_match


if __name__ == "__main__":
    interface = "galaxy"
    # # interface = "pptod"
    # if interface == "galaxy" :
    #     x = GalaxyInterface()
    #     exit()
    #     dial = x.get_dialogue("sng0073")
    #     print("len dial", len(dial))
    #     turn = x.get_turn_with_context(dial , 1)
    #     print("turn",turn)
    #     encoded = x.encode("sng0073",turn)
    #     print("encode",encoded)

    #     turn_output, bleu , tmp_dialog_result , pv_turn = x.predict_turn(encoded, 1)
    #     print("----bleu", bleu)
    #     print("---------------------------------")
    #     print("----turn_output", turn_output)
    #     print("---------------------------------")
    #     print("----tmp_dialog_result", tmp_dialog_result)
    #     print("---------------------------------")
    #     print("----pv_turn", pv_turn)
    #     print("---------------------------------")
    # else :
    #     with torch.no_grad():
    #         x = PptodInterface()
    #         data = x.data
    #         dialogue = x.get_dialogue("sng0073" , mode = 'dev')
    #         print("get dialogue done")
    #         turn = x.get_turn_with_context(dialogue , 1)
    #         print("turn with context", turn)
    #         # turn = x.get_turn(dialogue , 1)
    #         # print("turn", turn)
    #         print("get turn done")
            
    #         # input("fdbdsfbdsgbdgs")
    #         prepared_dial = x.prepare_turn(turn)
    #         print("preparing data done")
    #         print("prepared dial", prepared_dial)
    #         # import time
    #         # time.sleep(10)
    #         dial_result = x.predict_turn(prepared_dial)
    #         print("dial result", dial_result)
    #         dev_bleu, dev_success, dev_match = x.evaluate_turn(dial_result)
    #         print("dev_bleu",dev_bleu)
    #         print("dev_success",dev_success) 
    #         print("dev_match",dev_match)            
