import imp
import os
import random
from tqdm import tqdm
import yaml
import sys
import json
import numpy as np
import json
from torch.nn.utils import rnn
from RL4E2E.utils.constants import FRAMEWORK_PATH, GALAXY_PATH, MODELS_PATH, PPTOD_PATH
sys.path.append(PPTOD_PATH)
from reader import MultiWozReader
import ontology
import utils

from dataclasses import dataclass

from torch.nn.utils import rnn
import argparse
from Models.pptod.E2E_TOD.dataclass import MultiWozData
import copy

@dataclass 
class BPETextField : 
    vocab_path : str
    version : str
    data_root : str
    data_processed : str
    filtered : bool
    max_len : int
    min_utt_len : int
    max_utt_len : int
    min_ctx_turn : int
    max_ctx_turn : int
    tokenizer_type : str

@dataclass
class Data :
    data_dir : str
    data_name : str

@dataclass
class Trainer :
    seed : int
    gpu : int
    valid_metric_name : str
    num_epochs : int
    save_dir : str
    token_loss : bool
    batch_size : int
    log_steps : int
    valid_steps : int
    save_checkpoint : bool
    shuffle : bool
    sort_pool_size : int

@dataclass
class Model :
    init_checkpoint : str
    model : str
    num_token_embeddings : int
    num_pos_embeddings : int
    num_type_embeddings : int
    num_turn_embeddings : int
    num_act : int
    num_heads : int
    num_layers : int
    hidden_dim : int
    padding_idx : int
    dropout : float
    embed_dropout : float
    attn_dropout : float
    ff_dropout : float
    use_discriminator : bool
    dis_ratio : float
    bce_ratio : float
    pos_trainable : bool
    with_joint_act : bool
    with_rdrop_act : bool
    initializer_range : float
    lr : float
    weight_decay : float
    gradient_accumulation_steps : int
    warmup_steps : int
    max_grad_norm : float

@dataclass
class Gene:
    generator : str
    min_gen_len : int
    max_gen_len : int
    use_true_prev_bspn : bool
    use_true_prev_aspn : bool
    use_true_db_pointer : bool
    use_true_prev_resp : bool
    use_true_curr_bspn : bool
    use_true_curr_aspn : bool
    use_all_previous_context : bool
    use_true_bspn_for_ctr_eval : bool
    use_true_domain_for_ctr_eval : bool
    beam_size : int
    length_average : bool
    length_penalty : float
    ignore_unk : bool

@dataclass
class Hparams:
    do_train : bool
    do_test : bool
    do_infer : bool
    num_infer_batches : str
    hparams_file : str
    vocab_path : str
    version : str
    data_root : str
    data_processed : str
    filtered : bool
    max_len : int
    min_utt_len : int
    max_utt_len : int
    min_ctx_turn : int
    max_ctx_turn : int
    tokenizer_type : str
    BPETextField : BPETextField
    data_dir : str
    data_name : str
    Dataset : Data
    seed : int
    gpu : int
    valid_metric_name : str
    num_epochs : int
    save_dir : str
    token_loss : bool
    batch_size : int
    log_steps : int
    valid_steps : int
    save_checkpoint : bool
    shuffle : bool
    sort_pool_size : int
    Trainer : Trainer 
    init_checkpoint : str
    model : str
    num_token_embeddings : int
    num_pos_embeddings : int
    num_type_embeddings : int
    num_turn_embeddings : int
    num_act : int
    num_heads : int
    num_layers : int
    hidden_dim : int
    padding_idx : int
    dropout : float
    embed_dropout : float
    attn_dropout : float
    ff_dropout : float
    use_discriminator : bool
    dis_ratio : float
    bce_ratio : float
    pos_trainable : bool
    with_joint_act : bool
    with_rdrop_act : bool
    initializer_range : float
    lr : float
    weight_decay : float
    gradient_accumulation_steps : int
    warmup_steps : int
    max_grad_norm : float
    Model : Model 
    generator : str
    min_gen_len : int
    max_gen_len : int
    use_true_prev_bspn : bool
    use_true_prev_aspn : bool
    use_true_db_pointer : bool
    use_true_prev_resp : bool
    use_true_curr_bspn : bool
    use_true_curr_aspn : bool
    use_all_previous_context : bool
    use_true_bspn_for_ctr_eval : bool
    use_true_domain_for_ctr_eval : bool
    beam_size : int
    length_average : bool
    length_penalty : float
    ignore_unk : bool
    Generator : Gene
    use_gpu : bool





def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    for name in file_names:
        if name.startswith('epoch'):
            return name

def parse_config(version):
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
    parser.add_argument('--save_freq', default=1, help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save_dir', default=os.path.join(FRAMEWORK_PATH,"results"), help='Output directory.', type=str)
    # parser.add_argument('--render_freq', default=100, help='How often to render / save frames of an episode.', type=int)
    # parser.add_argument('--title', default="PDDQN", help="Prefix of output files", type=str)
    parser.add_argument('--action', default="train", help="train or evaluate", type=str)  
    parser.add_argument('--model', default="galaxy", choices=["galaxy", "pptod"], help="the model we want to test", type=str) 
    parser.add_argument('--version', default="2.0", choices=["2.0", "2.1"], help="the multiwoz version we want to use", type=str) 
    parser.add_argument('--num_selected_actions', default=3, help="how many actions to apply simultaniously", type=int) 
    if str(version) == "2.0":
        stream = open(os.path.join(MODELS_PATH,"pptod_config2.0.yaml"), 'r')
    if str(version) == "2.1":
        stream = open(os.path.join(MODELS_PATH,"pptod_config2.1.yaml"), 'r')
    args = yaml.load_all(stream, Loader=yaml.FullLoader)
    for doc in args:
        for key, value in doc.items():
            option = "--"+str(key)
            if type(value).__name__ == 'int' :
                parser.add_argument(option , type=int , help=option , default =value )  
            if type(value).__name__ == 'str' :
                if "/" in value:
                    parser.add_argument(option , type=str , help=option , default =os.path.join(MODELS_PATH,value))
                else :
                    parser.add_argument(option , type=str , help=option , default =value)


    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")

    parser.add_argument('--use_db_as_input', type=str, default='True', 
        help="True or False, whether includes db result as part of the input when generating response.")

    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')

    parser.add_argument('--train_data_ratio', type=float, default=1.0, help='the ratio of training data used for training the model')
    # # training configuration
    # parser.add_argument("--batch_size_per_gpu", type=int, default=1, help='Batch size for each gpu.')  
    # parser.add_argument("--number_of_gpu", type=int, default=1, help="Number of available GPUs.")  
    # model configuration
    
    return parser.parse_args()



all_sos_token_list = ['<sos_b>', '<sos_a>', '<sos_r>']
all_eos_token_list = ['<eos_b>', '<eos_a>', '<eos_r>']

class MyMultiWozData(MultiWozData):
    def __init__(self, model_name, tokenizer, cfg, data_path_prefix, shuffle_mode='shuffle_session_level', data_mode='train', use_db_as_input=True, add_special_decoder_token=True, train_data_ratio=1):
        self.use_db_as_input = use_db_as_input
        assert self.use_db_as_input in [True, False]
        self.add_special_decoder_token = add_special_decoder_token
        assert self.add_special_decoder_token in [True, False]

        self.cfg = cfg
        self.vocab = self._build_vocab(self.cfg)
        self.tokenizer = tokenizer
        self.special_token_list = self.add_sepcial_tokens()
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        self.reader = MultiWozReader(self.tokenizer, self.cfg, data_mode='test')

        # initialize bos_token_id, eos_token_id
        self.model_name = model_name
        if model_name.startswith('t5'):
            from transformers import T5Config
            t5config = T5Config.from_pretrained(model_name)
            self.bos_token_id = t5config.decoder_start_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
        else:
            raise Exception('Wrong Model Name!!!')
        self.bos_token = self.tokenizer.convert_ids_to_tokens([self.bos_token_id])[0]
        self.eos_token = self.tokenizer.convert_ids_to_tokens([self.eos_token_id])[0]

        self.all_sos_token_id_list = []
        for token in all_sos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_sos_token_id_list.append(one_id)

        self.all_eos_token_id_list = []
        for token in all_eos_token_list:
            one_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            self.all_eos_token_id_list.append(one_id)


        bs_prefix_text = 'translate dialogue to belief state:'
        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
        da_prefix_text = 'translate dialogue to dialogue action:'
        self.da_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(da_prefix_text))
        nlg_prefix_text = 'translate dialogue to system response:'
        self.nlg_prefix_id = self.tokenizer.convert_tokens_to_ids(tokenizer.tokenize(nlg_prefix_text))

        self.dev_json_path = data_path_prefix + '/multi-woz-fine-processed/multiwoz-fine-processed-dev.json'
        with open(self.dev_json_path) as f:
            self.dev_raw_data = json.load(f)
        
        self.test_json_path = data_path_prefix + '/multi-woz-fine-processed/multiwoz-fine-processed-test.json'
        with open(self.test_json_path) as f:
            self.test_raw_data = json.load(f)
    
    # def get_dialogue(self , dial_id , mode = 'test'):
    #     if mode == 'dev':
    #         data = self.dev_raw_data
    #     else :
    #         data = self.test_raw_data
    #     for dial in data :
    #         if dial[0]["dial_id"] == dial_id:
    #             return dial

    def prepare_dialogue(self ,dial):
        dial_id_list = self.tokenize_raw_data([dial])
        dial_list = self.flatten_data(dial_id_list)
        return dial_list

    def transform_turn(self , turn):
        turn_copy = copy.deepcopy(turn)
        turn_copy["usdx"] = self.get_utterance(turn)
        return turn_copy


    def build_all_evaluation_batch_list(self, ref_bs, ref_act, ref_db, input_contain_db, eva_batch_size , turn):
        '''
            bool ref_bs: whether using reference belief state to perform generation
                    if with reference belief state, then we also use reference db result
                    else generating belief state to query the db
            bool ref_act: whether using reference dialogue action to perform generation
                    if true: it always means that we also use reference belief state
                    if false: we can either use generated belief state and queried db result or
                              use reference belief state and reference db result
            str = test  eva_mode: 'dev' or 'test'; perform evaluation either on dev set or test set
           str = test  eva_batch_size: size of each evaluated batch
        '''
        all_bs_input_id_list, all_da_input_id_list, all_nlg_input_id_list, all_parse_dict_list = \
        [], [], [], []
        for item in [turn] :
            one_bs_input_id_list, one_da_input_id_list, one_nlg_input_id_list, one_parse_dict = \
            self.parse_one_eva_instance(item, ref_bs, ref_act, ref_db, input_contain_db)
            all_bs_input_id_list.append(one_bs_input_id_list)
            all_da_input_id_list.append(one_da_input_id_list)
            all_nlg_input_id_list.append(one_nlg_input_id_list)
            all_parse_dict_list.append(one_parse_dict)
        assert len(all_bs_input_id_list) == len(all_da_input_id_list)
        assert len(all_da_input_id_list) == len(all_nlg_input_id_list)
        assert len(all_nlg_input_id_list) == len(all_parse_dict_list)
        bs_batch_list = all_bs_input_id_list
        da_batch_list = all_da_input_id_list
        nlg_batch_list = all_nlg_input_id_list
        parse_dict_batch_list = all_parse_dict_list
        batch_num = len(bs_batch_list)
        
        final_batch_list = []
        for idx in range(batch_num):
            one_final_batch = [bs_batch_list[idx], da_batch_list[idx], nlg_batch_list[idx], parse_dict_batch_list[idx]]
            if len(bs_batch_list[idx]) == 0: 
                continue
            else:
                final_batch_list.append(one_final_batch)
        return final_batch_list


sys.path.append(GALAXY_PATH)

from galaxy.data.field import MultiWOZBPETextField
from galaxy.data.tokenizer import Tokenizer
from galaxy.utils.db_ops import MultiWozDB
import spacy


class MyMultiWOZBPETextField(MultiWOZBPETextField):
    """
    this class inherits from MultiWOZBPETextField which is the data reader class for multiwoz in galaxy
    it's purpose is to change the particular behaviour of loading and encoding the data since we want 
    to conserve the text data to apply nlp transformations on it 
    """
    def __init__(self, hparams):
        super(MultiWOZBPETextField, self).__init__(hparams)
        self.nlp = spacy.load('en_core_web_sm')

        self.db = MultiWozDB({
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        })
        self._build_vocab()

        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type=hparams.tokenizer_type)

        test_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'data/multiwoz{self.version}/testListFile.json'), 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'data/multiwoz{self.version}/valListFile.json'), 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        self._load_data(save_temp = False)

        return

    def _load_data(self, save_temp=True):
        """
        load processed data and encode, or load already encoded data
        """
        if save_temp:  # save encoded data
            encoded_file = os.path.join(self.data_root, f'data/multiwoz{self.version}', self.data_processed)

            if os.path.exists(encoded_file):
                self.data = json.loads(
                    open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
                encoded_data = json.loads(open(encoded_file, 'r', encoding='utf-8').read())
                self.train = encoded_data['train']
                self.dev = encoded_data['dev']
                self.test = encoded_data['test']
            else:
                # not exists, encode data and save
                self.data = json.loads(
                    open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
                self.train, self.dev, self.test = [], [], []
                for fn, dial in tqdm(self.data.items()):
                    if '.json' in fn:
                        fn = fn.replace('.json', '')
                    if self.dev_files.get(fn):
                        self.dev.append(self._get_encoded_data(fn, dial))
                    if self.test_files.get(fn):
                        self.test.append(self._get_encoded_data(fn, dial))
                    else:
                        self.train.append(self._get_encoded_data(fn, dial))

                # save encoded data
                encoded_data = {'train': self.train, 'dev': self.dev, 'test': self.test}
                json.dump(encoded_data, open(encoded_file, 'w'), indent=2)
        else:  # directly read processed data and encode
            self.data = json.loads(
                open(os.path.join(self.data_root, f'data/multiwoz{self.version}/data_for_galaxy.json'), 'r', encoding='utf-8').read().lower())
            self.train, self.dev, self.test = [], [], []
            for fn, dial in self.data.items():
                if '.json' in fn:
                    fn = fn.replace('.json', '')
                if self.test_files.get(fn):
                    self.test.append({fn : dial})
                if self.dev_files.get(fn):
                    self.dev.append({fn : dial})
                else:
                    self.train.append({fn : dial})

        random.shuffle(self.train)
        print('train size:{}, dev size:{}, test size:{}'.format(len(self.train), len(self.dev), len(self.test)))
