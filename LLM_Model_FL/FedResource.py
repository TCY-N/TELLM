import math
import numpy as np
import sys

class ResControl():
    def __init__(self,  env_args, epoch,weight_size, time_lim, noise, B,gamma) -> None:
        self.set_paprm(env_args)
        self.epoch = epoch 
        self.weight_size = weight_size
        self.time_lim = time_lim
        self.noise = noise
        self.noise_1 = 1/noise
        self.n = len(self.env)
        self.B =  B
        self.gamma = gamma
    
    def set_paprm(self,env_args):
        self.env = env_args

    def power_ctrl(self, env, f, bandwidth,):
        
        eta_n = env['G']/(self.noise*bandwidth)
        w_n = env['p_m']*self.weight_size/ (bandwidth*math.log2(1+eta_n*env['p_m']))
        time_com = bandwidth*(self.time_lim-self.epoch*env['cycle']*env['data_size']/f)
        thres = w_n*bandwidth*math.log2(math.e)/self.weight_size - 1/eta_n
        if time_com <= 0:
            return env['p_m']
        p_tilde = 1/eta_n * (2**(self.weight_size/(time_com))-1)
        # cmp = self.epoch*self.env[5]['cycle']*self.env[5]['data_size']/self.env[5]['f_max']
        # cmp1 = self.epoch*env['cycle']*env['data_size']/f
        
        step_size = 0.1

        for _ in range(50):
            if thres < p_tilde:
                p = p_tilde
            elif thres > env['p_m']:
                p = env['p_m']
            else:
                p = thres

            g_w = p*self.weight_size - w_n*(bandwidth*math.log2(1+eta_n*p))
            if g_w == 0:
                break
            else:
                d_n = g_w/(bandwidth*math.log2(1+eta_n*p))
                w_n = max(w_n + step_size*d_n,p*self.weight_size/(bandwidth*math.log2(1+eta_n*p)) )
        
        return p


    def cpu_ctrl(self, env, p, bandwidth):
        f_tilde = self.epoch * env['cycle'] * env['data_size']/(self.time_lim-(self.weight_size/(bandwidth*math.log2(1+p*env['G']/(self.noise*bandwidth)))))
        if f_tilde > env['f_max'] or f_tilde <= 0:
            return env['f_max']
        return max(f_tilde, env['f_min'])
    

    
    def time_cal(self,b,p,f):
        comm_time = [ self.weight_size/(b[i]*math.log2(1+p[i]*self.env[i]['G']/(self.noise*b[i]))) for i in range(self.n)]
        cmp_time = [self.epoch*self.env[i]['cycle']*self.env[i]['data_size']/f[i] for i in range(self.n)]
        return comm_time, cmp_time
    
    def energy_cal(self,b,p,f):
        comm_enrgy = [p[i]*self.weight_size/(b[i]*math.log2(1+p[i]*self.env[i]['G']/(self.noise*b[i]))) for i in range(self.n)]
        cmp_energy = [self.epoch*self.env[i]['cycle']*self.env[i]['data_size']*self.gamma*f[i]**2 for i in range(self.n)]
        return comm_enrgy, cmp_energy
    
    def val_cal(self,b,p,f):
        comm_enrgy, cmp_energy = self.energy_cal(b,p,f)
        comm_time, cmp_time = self.time_cal(b,p,f)
        energy = np.sum( [comm_enrgy[i] + cmp_energy[i] for i in range(self.n)] )
        time = np.max([comm_time[i] + cmp_time[i] for i in range(self.n)])
        return energy, time

    def ALTD(self, env, f_1, b=-1):
        f_0 = env['f_min']
        f = env['f_max']
        if b == -1:
            b = self.B/self.n
        for _ in range(50):
            p = self.power_ctrl(env, f, b)
            f = self.cpu_ctrl(env, p, b)
            if f_0 == f_1:
                break
            f_0 = f_1
            f_1 = f
        return p, f


    def BFPA1(self):
        i = 0
        p = [self.env[i]['p_m'] for i in range(self.n)]
        f = [self.env[i]['f_max'] for i in range(self.n)]
        b = [self.B/self.n for _ in range(self.n)]
        while i < 50:
            b = self.bandwidth_ctrl(p, f)
            flag = 1
            for j in range(self.n):
                f_1 = f[j]
                p[j], f[j] =  self.ALTD(self.env[j], f_1, b=b[j])
                if f_1 != f[j]:
                    flag = 0
            if flag:    
                break
            # b = self.bandwidth_ctrl(p, f)
            energy, time = self.val_cal(b,p,f)
            print("iter{} energy:{}, time:{}".format(i,energy,time))
            i+=1
        return b,p,f
       

if __name__=="__main__":
    np.random.seed(10)
    epoch = 5
    weight_size = 1e8#/(1e6)
    Bandweith = 2e7#/(1e6)
    cycle = 800
    time_lim = 350
    number = 10
    All_clients_dataset_info = np.random.uniform(2.0e6, 1.0e7, size = (1, number))
    All_clients_transmission_power_info = np.random.uniform(0.001, 1, size = (1, number))
    G = np.random.uniform(1.0e-3, 1.0e-2, size = (1, number))
    f = np.random.uniform(0.5,4.0, size = (1, number))*1.0e+09
    noise = 1e-15#*(1e6)
    gamma = 1e-28
    envs = []
    for i in range(number):
        env = {}
        env['data_size'] = All_clients_dataset_info[0][i]
        env['p_m'] = All_clients_transmission_power_info[0][i]
        env['f_max'] = f[0][i]
        env['f_min'] = 1e4
        env['G'] = G[0][i]
        env['cycle'] = cycle
        envs.append(env)

    allocator = ResControl( envs, epoch, weight_size, time_lim, noise,Bandweith,gamma )
    

    energy, time = allocator.val_cal([Bandweith/number for _ in range(number)],All_clients_transmission_power_info[0],
                                     f[0])
    print("Max: energy:{}, time:{}".format(energy,time))

    p_td = [0 for _ in range(number)]
    f_td = [0 for _ in range(number)]
    b_td = [Bandweith/number for _ in range(number)]
    for i in range(number):
        p_td[i], f_td[i] = allocator.ALTD(envs[i], f[0][i], Bandweith/number)
    energy, time = allocator.val_cal(b_td,p_td,f_td)
    print("ALTD energy:{}, time:{}".format(energy,time))





    
