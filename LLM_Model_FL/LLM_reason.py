import os
from config import get_args
args = get_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA 
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import random
from copy import deepcopy
import json
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object
import matplotlib.pyplot as plt
import scipy.io as scio

from server import *
from ft_test import LlamaModel

 


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_fl(env_index, seed, alg_seed):
    envserver.reset(env_index, seed)

    global_parameters = {}
    for key, var in envserver.net.state_dict().items():
        global_parameters[key] = var.clone()

    state = []

    for i in range(args['num_of_clients']):
        state.append(envserver.init_static_state(i))

    if args['select'] == 'grpo' or args['select'] == 'ppo':
        ref = '[1 2 3 4]'
        last_inf = ''
    seed_everything(alg_seed)
    # envserver.seed_everything(int(time.time()))

    return state, global_parameters, ref, last_inf

def single_run(llama, command, step, state, global_parameters, ref, temperature=1.0, last_inf=''):
    print(f"communicate round {step}, {command}")
    
    for index in range(args['num_of_clients']):
        envserver.update_state_broadcast(state[index], index, global_parameters)
 
    if args['select'] == 'grpo' or args['select'] == 'ppo':
        
        # Adjusting the randomness of temperature control
        # if step <= 20:
        #     temperature = 1+(2-1)/20*step
        # elif step <= 40:
        #     temperature = 2+(6-2)/20*(step-20)
        # elif step <= 60:
        #     temperature = 6+(6-5.5)/40*(step-20)
        # else:
        #     temperature = 5.5

        envserver.num_in_comm = command['K']  # number of clients in communication
        llama.args['sele_clients'] = command['K']
        args['num_of_clients'] = command['M']   # total number of clients
        llama.args['num_of_clients'] = command['M']
        llama.allowed_token_ids = llama._limit_token()

        args['time_lim'] = command['qos_time']  # Qos for time
        inputs = llama.preprocess(last_inf, step, state, args['num_of_clients'], envserver.num_in_comm, args['time_lim'], ref, args['llm_model'])
        print(f"inputs: {inputs}")

        outputs = llama.generate(inputs,temperature)

        sentence = llama.tokenizer.convert_ids_to_tokens( outputs[0], skip_special_tokens=True)
        action = [int(c) for c in sentence if c.isdigit()]
        select_client = action
    elif args['select'] == 'random':  
        select_client= random.sample(range(args['num_of_clients']),args['sele_clients'])


    p, f, comm_energy, cmp_energy, comm_time, cmp_time = envserver.get_res(select_client)
    energy, timea = envserver.get_time_energy(comm_energy, cmp_energy, comm_time, cmp_time)

        # if action != None and energy<3:
        #     select_client = action
        #     break
    clients_in_comm = ['client{}'.format(i) for i in select_client]
    acc = envserver.step(global_parameters, clients_in_comm)

    last_inf =f"For last round, the select clients are {select_client}, and the time consumption is {timea}, the energy consumption is {energy}, the accuracy of global model is {acc:.4f}. "

    reward = envserver.get_reward(energy, timea, acc)
    round_action = [select_client] + [ {'client{}'.format(client):{'power':p[client], 'frequency':f[client]}} for i, client in 
    enumerate(select_client)]

    print(f'In communication round {step}, select clients: {select_client}, reward: {reward}, time: {timea}, energy: {energy}, acc: {acc}')
    print('=====================================================')

    return select_client, state, acc, p, f, energy, timea, last_inf, global_parameters
    
def init_env(dev):
    datafile = 'original_data_1000_{}.mat'.format(args['num_of_clients'])
    env_config  = scio.loadmat(datafile)
    envserver = Env_server(args,  env_config)
    max_episodes = args['episode']
    max_steps = args['num_comm']


    print(args['select'])
    
    if args['llm_model'] == 'llama':
        MODEL_PATH = "../LLM-Research/Llama-3___2-1B-Instruct"

    if args['select'] == 'grpo':
        lora_path = './model_check/checkpoint-1000g20model'
    elif args['select'] == 'ppo':
        lora_path = './model_check/checkpoint-ppo'

    if args['select'] == 'grpo' or args['select'] == 'ppo':
        llama = LlamaModel(MODEL_PATH,lora_path, dev, args)
    else:
        llama = None
    
    return envserver, max_steps, llama


if __name__ == "__main__":
    args = args.__dict__
    seed_everything()
    dev = 'cuda:'+args['gpu']
    envserver, max_steps, llama = init_env(dev)
    episode = [0] # index for environment
    seed_episode = [42] # seed for environment
    alg_seed = [7] # seed for algorithm randomness

    for index, eps in enumerate(episode):
        state, global_parameters, ref, last_inf = init_fl(eps, seed_episode[index], alg_seed[index])

        for step in range(1,max_steps+1):
            command = {'M':20, 'K':4, 'qos_time':20}
            global_parameters_bef = deepcopy(global_parameters)
            select_client, state, acc, p, f, energy, timea, last_inf, global_parameters = single_run(llama, command, step, state, global_parameters, ref, temperature=1.0, last_inf=last_inf)

            global_aft_weight_params = envserver.flatten_and_concat_params(global_parameters, dev)
            global_bef_weight_params = envserver.flatten_and_concat_params(global_parameters_bef, dev)
            gradient_global = envserver.get_gradient(global_aft_weight_params, global_bef_weight_params, args['learning_rate'])
            for index in select_client:
                envserver.update_state_agg(state[index], index, global_bef_weight_params, gradient_global, step)
