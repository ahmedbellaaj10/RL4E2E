import os
import yaml

def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    for name in file_names:
        if name.startswith('epoch'):
            print (name)
            return name

def parse_config():
    parser = argparse.ArgumentParser()
    stream = open("/home/ahmed/RL4E2E/Models/pptod_config.yaml", 'r')
    args = yaml.load_all(stream, Loader=yaml.FullLoader)
    for doc in args:
        for key, value in doc.items():
            print ()
            option = "--"+str(key)
            if type(value).__name__ == 'int' :
                parser.add_argument(option , type=int , help=option , default =value )  
            if type(value).__name__ == 'str' :
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



import argparse