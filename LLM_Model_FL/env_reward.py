import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from DT_ModelBased.decision_transformer.models.decision_transformer import DecisionTransformer

from server import Env_server
from FedResource import ResControl
import numpy as np
import re
import copy

class RewardModel():
    def __init__(self, config, variant):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = variant['state_dim']
        self.loss_dim=variant['loss_dim']
        self.act_dim=variant['act_dim']
        self.model = DecisionTransformer(
            state_dim=variant['state_dim'],
            loss_dim=variant['loss_dim'],
            act_dim=variant['act_dim'],
            max_length=variant['K'],
            # max_ep_len=variant['max_ep_len'],
            max_ep_len=101,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
        self.model.to(self.device)
        if self.config.DiSigma == 0.06:
            checkpoint = torch.load('../LLM_simu/checkpoint_99.pth',map_location=self.device)
        else:
            checkpoint = torch.load('../LLM_simu/checkpoint_99_Di15.pth',map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mask = torch.ones((self.state_dim), dtype=torch.bool)
        self.mask[0::4] = False

    def compute_reward(self, state, loss, action, reward, timesteps):
        """计算奖励值"""
        action = torch.cat([action, torch.zeros(( 1, self.act_dim), device=self.device)], dim=0)
        reward = torch.cat([reward, torch.zeros(( 1, 1), device=self.device)])
        state = torch.cat([state, torch.zeros(( 1, self.state_dim), device=self.device)])
        loss = torch.cat([loss, torch.zeros(( 1, self.loss_dim), device=self.device)])
        self.model.eval()
        state_preds, loss_preds, reward_preds = self.model.get_states(state, loss, action, reward, timesteps)
        state[-1, self.mask] = state_preds[0,:]
        state[-1, ~self.mask] = state[0, ~self.mask]
        loss[-1] = loss_preds
        reward[-1] = reward_preds

        return state, loss, reward
    
class Fl_env(Env_server):
    
    def get_init_state(self):
        global_parameters = {}
        for key, var in self.net.state_dict().items():
            global_parameters[key] = var.clone()
        state = []
        for i in range(self.args['num_of_clients']):
            state.append(self.init_static_state(i))
        for index in range(self.args['num_of_clients']):
            self.update_state_broadcast(state[index], index, global_parameters)
        acc = self.valid( global_parameters,self.testDataLoader).cpu().numpy()
        
        return state, acc

    def init_state_txt(self, state, acc):
        last_inf = f'The initial global model accuracy is {acc}'
        actions_history = None
        ref = '[1 2 3 4]'
        inputs = self.process2txt( last_inf, 1, state, self.args['num_of_clients'], self.num_in_comm, self.args['time_lim'], actions_history, ref)
        return inputs
    
    def get_state_model(self, data_states,acc): 
        state_model = []
        loss_model = []
        acc_model = acc
        for data_state in data_states:
            client_state = []
            data_size = data_state['data_size']/60000
    
            loss_traind = data_state['loss_trained']
            
            if loss_traind!=0:
                loss_traind = loss_traind/data_state['data_size'] # 重新计算loss
                loss_traind = np.exp2(-loss_traind)
            
            client_state.append(data_size)
            client_state.append(loss_traind)
            # client_state.append(data_state['disent'])
            client_state.append(data_state['inner_product'])
            client_state.append(data_state['sign'])

            loss = data_state['loss']/data_state['data_size']
            loss = np.exp2(-loss)

            state_model.append(client_state)
            loss_model.append(loss)

        return state_model, loss_model, acc_model

    def dats_list2dic(self, state_list, loss_list, state):
        for i, val in enumerate(state):
            val['loss'] = -np.log2(loss_list[i])* val['data_size']
            val['loss_trained'] = 0 if state_list[i*4+1]==0 else -np.log2(state_list[i*4+1])* val['data_size']
            val['inner_product'] = state_list[i*4+2]
            val['sign'] = state_list[i*4+3]
        return state
    
    def val_form(self, data):
        if data != 0:
            if self.args['llm_model'] == 'gpt':
                return f'{data:.2f}'
            else:
                return f'{data:.4f}'
        
        return '\\'
    
    def val_form_int(self, data):
        if data != 0:
            return f'{data}'
        return '\\'

    def process2txt(self, last_inf, round, state, client_num, comm_num, qos,  ref, band=2e7):
        '''data表示一个step的数据, returns为上一轮反馈结果,round从1开始,0为开始之前'''
        input = "As a federated training agent, you are responsible for selecting the most suitable clients from the device pool to optimize performance in the current round.\n"
        input += f"Select {comm_num} indexes from given {client_num} clients to participate in federated learning training based on the return information in last round and the clients\' states.\n"
        round_inf = f"The total bandwidth of the system is {band:.2e}. The reference QoS time is {qos:.2f}s. The current communication is round {round}.\n"
        input += last_inf + round_inf
        input += 'The states information of each client is shown in the following table\n'

        # round_inf = f"The current communication is round {round}.\n"
        # input += "For last round, " + last_inf+round_inf

        state_inf = ''
        if self.args['llm_model'] == 'llama':
            input_head = "| client index | max power | max frequency | channel gain | number of compute cycles | data size | local loss | local loss after trained | inner-product betweent local model and global | percentage of same sign betweent local model and global model | last selected round | selected times |\n| --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            for key, val in enumerate(state):
                state_inf += f"| {key} |"
                # state_inf += f"the last selected round is {val['round']}, "
                state_inf += f" {val['p']:.6f} |"       # power
                state_inf += f" {val['f']:.4f} |"       # frequency
                state_inf += f" {val['G']:.4e} |"       # channel gain
                state_inf += f" {val['cycle']:.0f} |"   # compute cycle
                state_inf += f" {val['data_size']:.0f} |"   # data_size
                state_inf += f" {val['loss']:.4f} |"        # loss
                # state_inf += f" {val['loss_trained']} |" # loss_trained
                # state_inf += f" {val['inner_product']} |" # inner_product
                # state_inf += f" {val['sign']} |\n"       # sign
                state_inf += f" {self.val_form(val['loss_trained'])}"
                state_inf += f" {self.val_form(val['inner_product'])} |"
                state_inf += f" {self.val_form(val['sign'])} |"
                state_inf += f" {self.val_form_int(val['round'])} |"
                state_inf += f" {self.val_form_int(val['times'])} |\n"
                # state_inf += f"the selected times is {val['times']}, "
                # state_inf += f"and the disent to the global model is {val['disent']}.\n"
            state_inf += '\n'

        notice = f"Please select {comm_num} indexes from given {client_num}  client indexes as a result.\n"
        format_ref = f'Give one result directly. Do not ouput anything else. Your response consists only of indexes.'# Ensure that your response is concise and brief and only contains one indexes list.\nFormat of respond: {ref}.\n'
        input += input_head + state_inf + notice + format_ref 

        return input
    
    def compute_reward_online(self, action, global_parameters, state, step):
        """online"""
        global_parameters_bef = copy.deepcopy(global_parameters)
        select_client = action
        clients_in_comm = ['client{}'.format(i) for i in select_client]
        acc = self.step(global_parameters, clients_in_comm)
        for index in range(self.args['num_of_clients']):
            self.update_state_broadcast(state[index], index, global_parameters)

        global_aft_weight_params = self.flatten_and_concat_params(global_parameters, self.dev)
        global_bef_weight_params = self.flatten_and_concat_params(global_parameters_bef, self.dev)
        gradient_global = self.get_gradient(global_aft_weight_params, global_bef_weight_params, self.args['learning_rate'])
        for index in select_client:
            self.update_state_agg(state[index], index, global_bef_weight_params, gradient_global, step)

        return state, acc

class FL_reward():
    def __init__(self, env_args, env_config):
        self.env = Fl_env(vars(env_args), env_config)
        self.model_based = RewardModel(env_args, vars(env_args))

    def cal_reward1(self, acc, time, energy):
        # reward = delta_acc*acc/energy*1e4
        reward = acc/energy*1e2
        penity = time - self.env.args['time_lim']
        if penity > 0:
            reward = reward*(self.env.args['time_lim']/time)**6
        return reward
    
    def cal_reward(self, acc, time, energy, delta_acc):
        reward = (acc / energy *0.9 + acc*0.1)*1e2
        # reward = acc/energy*1e2
        penity = time - self.env.args['time_lim']
        if penity > 0:
            reward = reward*(self.env.args['time_lim']/time)**6
        return reward

    def sequence_reward(self, sentence):
        pattern = r'^[^\[\]]*\[\s*'  
        pattern += r'(?:0|[1-9]\d*)(?:\s* \s*(?:0|[1-9]\d*))*'  
        pattern += r'\s*\][^\[\]]*$'  
        if not re.fullmatch(pattern, sentence):
            return -0.1, None

        reward_item=[]
        success = True
        list_parttern = r'\[\s*' + r'(?:0|[1-9]\d*)(?:\s* \s*(?:0|[1-9]\d*))*' + r'\s*\]'
        match = re.findall(list_parttern, sentence)
        elements = list(map(int, match[0][1:-1].split()))

        if len(elements) != len(set(elements)):
            success = False
        else: reward_item.append(0.1)

        if len(elements) != self.env.num_in_comm:
            success = False
        else: reward_item.append(0.1)

        if any(not (0 <= x < self.env.args['num_of_clients']) for x in elements):
            success = False
        else: reward_item.append(0.1)
        
        if success:
            return sum(reward_item), elements
        else:
            return sum(reward_item), None

    def get_reward_state(self, data_states, state_model, loss_model, actions, acc_model, timesteps, round):
        
        state, loss, acc_reward = self.model_based.compute_reward( state_model, loss_model, actions, acc_model,timesteps)
        data_states = self.env.dats_list2dic(state[-1,:].cpu().numpy(), loss[-1,:].cpu().numpy(), data_states)

        # p, f, comm_energy, cmp_energy, comm_time, cmp_time = self.env.get_res([i for i in range(self.env.args['num_of_clients'])])
        select_client = [index for index, value in enumerate(actions[-1]) if value != 0]
        for i in select_client:
            data_states[i]['round'] = round
            data_states[i]['times'] += 1

        p, f, comm_energy, cmp_energy, comm_time, cmp_time = self.env.get_res([i for i in select_client])
        
        energy, time = self.env.get_time_energy(comm_energy, cmp_energy, comm_time, cmp_time)
        acc = acc_reward[-1,-1].cpu().numpy().item()
        # reward = self.cal_reward(acc_reward[-1,-1].cpu().numpy().item(), time, energy,round, delta_acc)
        for i in range(self.env.args['num_of_clients']):  
            self.env.get_state(data_states[i], i)

        acc_last = acc
        last_inf =f"For last round, the select clients are {select_client}, and the time consumption is {time}, the energy consumption is {energy}, the accuracy of global model is {acc_last:.4f}. "
        ref = '[1 2 3 4]'
        band = self.env.args['B']
        next_prompt = self.env.process2txt(last_inf, round+1, data_states, self.env.args['num_of_clients'], self.env.num_in_comm, self.env.args['time_lim'], ref, band)   ######

        return acc, time, energy, data_states, state, loss, acc_reward, next_prompt

    def get_reward_state_online(self, data_states,  action, global_parameters, round):
        
        data_states, acc_reward = self.env.compute_reward_online(action, global_parameters, data_states, round)

        # p, f, comm_energy, cmp_energy, comm_time, cmp_time = self.env.get_res([i for i in range(self.env.args['num_of_clients'])])
        select_client = action
        for i in select_client:
            data_states[i]['round'] = round
            data_states[i]['times'] += 1
        p, f, comm_energy, cmp_energy, comm_time, cmp_time = self.env.get_res([i for i in select_client])
        
        energy, time = self.env.get_time_energy(comm_energy, cmp_energy, comm_time, cmp_time)
        
        # reward = self.cal_reward(acc_reward.cpu().numpy().item(), time, energy,round)
        for i in range(self.env.args['num_of_clients']):  
            self.env.get_state(data_states[i], i)

        acc_last = acc_reward
        last_inf =f"For last round, the select clients are {select_client}, and the time consumption is {time}, the energy consumption is {energy}, the accuracy of global model is {acc_last:.4f}. "
        ref = '[1 2 3 4]'
        next_prompt = self.env.process2txt(last_inf, round+1, data_states, self.env.args['num_of_clients'], self.env.num_in_comm, self.env.args['time_lim'],  ref)   ######

        return acc_reward, time, energy, data_states, next_prompt  

        

