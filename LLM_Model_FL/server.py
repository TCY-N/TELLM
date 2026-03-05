import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
import random, math
from copy import deepcopy

from FedResource import ResControl
import gc

class Env_server():
    def __init__(self, args, env_config) -> None:
        self.args = args
        # self.mac = mac 
        self.num_in_comm = self.args['sele_clients']
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._getnet()
        self._getOp()
        self._getClients()
        self.env_config = env_config    
        self.t = 0
        self.client_env = []
        self.cycle = 7e5
        self.bit = 32

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

    def _getnet(self):
        self.net = None
        if self.args['model_name'] == 'mnist_2nn':
            self.net = Mnist_2NN()
        elif self.args['model_name'] == 'mnist_cnn':
            self.net = Mnist_CNN()

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.net = torch.nn.DataParallel(self.net)       
        self.net = self.net.to(self.dev)


    def _getOp(self):
        self.loss_func = F.cross_entropy
        if self.args['optimizer'] == 'SGD':
            self.opti = optim.SGD(self.net.parameters(), lr=self.args['learning_rate'])

    def _getClients(self):
        self.myClients = ClientsGroup('mnist', self.args['IID'], self.args['num_of_clients'], self.dev, self.args['uniform'], self.args['DiSigma'])
        self.testDataLoader = self.myClients.test_data_loader

    def reset(self,t,seed):
        # Reinitialize network parameters
        self.seed_everything(seed)
        self._getClients()
        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.net.apply(reset_weights)
        
        self.Gaussian_noise = 10**(-17.4)
        self.epoch = self.args['epoch'] 
        self.weight_size = sum(p.numel() for p in self.net.parameters()) * self.bit
        self.time_lim = self.args['time_lim']
        # self.noise = 10**(-17.4)
        self.B =  self.args['B']
        self.gamma = 1e-28
        self.client_env = []
        self.G = deepcopy(self.env_config['G'][t])   

        for i in range(self.args['num_of_clients']):
            env = {}
            env['weight_size'] = self.weight_size
            env['data_size'] = len(self.myClients.clients_set['client{}'.format(i)].train_ds)
            env['p_m'] = self.env_config['All_clients_transmission_power_info'][t][i]
            env['f_max'] = self.env_config['f'][t][i]
            env['f_min'] = 0
            env['G'] = self.env_config['G'][t][i]
            env['cycle'] = self.cycle
            env['bandw'] = self.args['B']/self.num_in_comm
            self.client_env.append(env)
        

    def get_res(self,select_clients):
        envs = []
        for i in select_clients:
            envs.append(self.client_env[i])

        allocator = ResControl( envs, self.args['epoch'], self.weight_size, self.args['time_lim'], self.Gaussian_noise,self.args['B'], self.gamma )
        p_td = [0 for _ in range(len(select_clients))]
        f_td = [0 for _ in range(len(select_clients))]
        b_td = [envs[i]['bandw'] for i in range(len(select_clients))]
        for i in range(len(select_clients)):
            p_td[i], f_td[i] = allocator.ALTD(envs[i], envs[i]['f_max'], envs[i]['bandw'])

        comm_enrgy, cmp_energy = allocator.energy_cal(b_td,p_td,f_td)
        comm_time, cmp_time = allocator.time_cal(b_td,p_td,f_td)
        p, f = {}, {}
        for i, val in enumerate(select_clients):
            p[val] = p_td[i]
            f[val] = f_td[i]
        
        return p, f, comm_enrgy, cmp_energy, comm_time, cmp_time
    
    def get_max_comsump(self):
        envs = []
        for i in range(self.args['num_of_clients']):
            envs.append(self.client_env[i])
        allocator = ResControl( envs, self.args['epoch'], self.weight_size, self.args['time_lim'], self.Gaussian_noise,self.args['B'], self.gamma )
        p_td = [self.client_env[i]['p_m'] for i in range(self.args['num_of_clients'])]
        f_td = [self.client_env[i]['f_max'] for i in range(self.args['num_of_clients'])]
        b_td = [self.client_env[i]['bandw'] for i in range(self.args['num_of_clients'])]
        comm_enrgy, cmp_energy = allocator.energy_cal(b_td,p_td,f_td)
        comm_time, cmp_time = allocator.time_cal(b_td,p_td,f_td)

        max_clients_energy = [comm_enrgy[i] + cmp_energy[i] for i in range(self.args['num_of_clients'])] 
        min_clients_time = [comm_time[i] + cmp_time[i] for i in range(self.args['num_of_clients'])]
        return max_clients_energy, min_clients_time
    
    def get_time_energy(self, comm_enrgy, cmp_energy, comm_time, cmp_time):
        energy = np.sum( [comm_enrgy[i] + cmp_energy[i] for i in range(self.num_in_comm)] )
        time = np.max([comm_time[i] + cmp_time[i] for i in range(self.num_in_comm)])
        return energy, time


    def get_time_cmp(self, env, f):
        cpu_freq = f
        data_set = env['data_size']
        time_train = self.args['epoch'] * env['cycle'] * data_set/ cpu_freq
        return time_train

    def get_time_com(self, env, p):
        bandwidth = env['bandw']

        channel_gain = env['G']
        time_up = self.weight_size/ bandwidth * np.log2(
            1 + (p * channel_gain) / (self.Gaussian_noise * bandwidth))
        return time_up

    def get_energy_cmp(self, env, f):
        data_set = env['data_size']
        energy_cmp = self.args['epoch'] * self.gamma * env['cycle'] * data_set * (f ** 2)
        return energy_cmp

    def get_energy_com(self, time, p):
        energy = p * time
        return energy

    def flatten_and_concat_params(self, parameters, device):
        if parameters == None:
            return None
        with torch.no_grad():
            flattened_params = torch.cat([p.view(-1) for p in parameters.values()]).to(device)
        return flattened_params

    def get_loss(self, client, global_parameters):
        loss =  client.get_loss(self.args['batchsize'], self.net, self.loss_func, global_parameters)
        return loss
    
    def get_loss_trained(self, client):
        loss =  client.get_loss(self.args['batchsize'], self.net, self.loss_func, client.local_parameters)
        return loss

    def get_disten(self, client, global_parameters):
        disten = 0
        if client.local_parameters == None:
            return -1
        else:
            with torch.no_grad():
                params1 = torch.cat([p.view(-1) for p in client.local_parameters.values()]).to(self.dev)
                params2 = torch.cat([p.view(-1) for p in global_parameters.values()]).to(self.dev)
                disten = torch.sum((params1 - params2)**2)
            # print(disten)
        return disten.cpu().data.numpy()

    def get_disten_new(self, params1, params2):
        with torch.no_grad():
            disten = torch.sum((params1 - params2)**2)
            # print(disten)
        return disten.cpu().data.numpy()

    def get_inner_product(self, params1, params2):
        with torch.no_grad():                
            norm1 = torch.norm(params1)
            norm2 = torch.norm(params2)

            dot_product = torch.dot(params1, params2)

            if norm1 == 0 or norm2 == 0:
                normalized_inner_product = 0.0
            else:
                normalized_inner_product = dot_product / (norm1 * norm2)
            # print(disten)
        return normalized_inner_product.cpu().data.numpy()

    def get_sign(self, params1, params2):
        with torch.no_grad():
            sign1 = torch.sign(params1)
            sign2 = torch.sign(params2)

            same_sign_count = (sign1 == sign2).sum().item()

            total_params = params1.numel()

            same_sign_percentage = (same_sign_count / total_params)
        return same_sign_percentage

    def get_gradient(self, params1, params2, lr):
        with torch.no_grad():
            difference = params1 - params2
            difference /= lr
        return difference

    def get_norm(self, difference):
        with torch.no_grad():
            res = torch.norm(difference)
        return res.cpu()


    def get_reward(self, energy, time, acc):
        weight_time = 1
        weight_energy = 1
        if energy > self.args['energy_lim']:
            weight_energy = pow(self.args['energy_lim']/energy,self.args['beta'])
        if time > self.args['time_lim']:
            weight_time = pow(self.args['time_lim']/time,self.args['alpha'])
        reward = acc*weight_time*weight_energy
        return reward#.item()

    def valid(self, global_parameters,vailDataLoader):
        with torch.no_grad():
            self.net.load_state_dict(global_parameters, strict=True)
            sum_accu = 0
            num = 0
            for data, label in vailDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
                acc_f = sum_accu / num
        return acc_f
    
    def get_tensors_on_gpu(self):
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                if obj.is_cuda: 
                    print(f"Tensor on GPU -> Shape: {obj.size()}, Device: {obj.device}, Dtype: {obj.dtype}")
    

    def get_state(self, states, index):
        states['p'] = self.client_env[index]['p_m']
        states['f'] = (self.client_env[index]['f_max'])
        rng = np.random.RandomState()
        states['G'] = (self.G[index])*(1+ abs(rng.normal(0.1, 1)))
        self.client_env[index]['G'] = states['G']
        states['cycle'] = (self.client_env[index]['cycle'])

        
        return states


    def init_static_state(self,index):
        states = {}
        states['loss'] = 0
        states['times'] = 0
        states['round'] = 0
        states['data_size'] = self.client_env[index]['data_size']
        # states['disent'] = 0
        states['loss_trained'] = 0
        states['inner_product'] = 0
        states['sign'] = 0
        # states['norm'] = 0
        states['p'] = self.client_env[index]['p_m']
        states['f'] = (self.client_env[index]['f_max'])
        states['G'] = self.G[index]
        states['cycle'] = self.client_env[index]['cycle']
        states['acc'] = 0
        return states

    def update_state_broadcast(self, states, index, global_parameters):
        states['loss'] = (self.get_loss(self.myClients.clients_set['client{}'.format(index)], global_parameters))
        

    def update_state_agg(self, states, index, global_bef_weight_params, gradient_global, step):
        local_weight_params = self.flatten_and_concat_params(self.myClients.clients_set['client{}'.format(index)].local_parameters, self.dev)

        gradient_local = self.get_gradient( local_weight_params, global_bef_weight_params,self.args['learning_rate'])
        # gradient_global = self.get_gradient(global_aft_weight_params, global_bef_weight_params)
        # states['norm'] = self.get_norm(gradient_local)
        states['inner_product'] = self.get_inner_product(gradient_local, gradient_global)
    
        # states['disent'] = self.get_disten_new(local_weight_params, global_bef_weight_params)
        states['sign'] = self.get_sign(gradient_local, gradient_global)
        states['round'] = step
        states['times'] += 1
        states['loss_trained'] = self.get_loss_trained(self.myClients.clients_set['client{}'.format(index)])



    def select_clients(self, select, state, time, energy):    
        select_client = select.probing_oort(state['loss'], self.args['num_client'], self.num_in_comm, time, energy, 20)
        return select_client


    def step(self,global_parameters,clients_in_comm):
        '''Run FL, earn rewards, detect terminated state'''
        sum_parameters = None
        sum_data = 0
        for client in clients_in_comm:

            local_parameters = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                         self.loss_func, global_parameters, self.opti)
            client_sata_size = len(self.myClients.clients_set[client].train_ds)
            sum_data += client_sata_size
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone() * client_sata_size

            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var] * client_sata_size

        for var in global_parameters:
            # global_parameters[var] = (sum_parameters[var] / self.num_in_comm)
            global_parameters[var] = (sum_parameters[var] / sum_data)

        acc = self.valid( global_parameters,self.testDataLoader)

        return acc.item()

