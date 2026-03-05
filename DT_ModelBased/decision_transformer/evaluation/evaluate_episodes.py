import numpy as np
import torch
import json
# import copy 

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        loss_dim,
        model,
        max_ep_len=101,
        device='cuda',
        path_name=None,
    ):

    fold_name = path_name+'/eval_teacher.txt'

    ans = []
    
    model.eval()
    model.to(device=device)

    state_target = torch.from_numpy(env['state']).reshape(max_ep_len ,state_dim).to(device=device, dtype=torch.float32)
    actions = torch.from_numpy(env['action']).reshape(max_ep_len, act_dim).to(device=device, dtype=torch.float32)
    reward_target = torch.from_numpy(env['acc']).reshape(max_ep_len, 1).to(device=device, dtype=torch.float32)
    losses_target = torch.from_numpy(env['loss']).reshape(max_ep_len, loss_dim).to(device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


    state = torch.zeros((1, state_dim), device=device, dtype=torch.float32)
    state[-1]  = state_target[0,:]
    loss = torch.zeros((1, loss_dim), device=device, dtype=torch.float32)
    loss[-1]  = losses_target[0,:]
    reward = torch.zeros((1, 1), device=device, dtype=torch.float32)
    reward[-1]  = reward_target[0,:]

    mask = torch.ones((state_dim), dtype=torch.bool)
    mask[0::4] = False

    loss_error, state_error, reward_error = [],[],[]

    for t in range(max_ep_len-1):
        # add padding
        action = actions[0:t+1].to(dtype=torch.float32)
        action = torch.cat([action, torch.zeros((1, act_dim), device=device)], dim=0)
        reward = torch.cat([reward, torch.zeros((1,1), device=device)])
        state = torch.cat([state, torch.zeros((1, state_dim), device=device)])
        loss = torch.cat([loss, torch.zeros((1, loss_dim), device=device)])

        state_preds, loss_preds, reward_preds = model.get_states(
            state.to(dtype=torch.float32),
            loss.to(dtype=torch.float32),
            action.to(dtype=torch.float32), 
            reward.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
         
        ac_mask = torch.repeat_interleave(action[-2,], repeats=3,dim=0).bool()
        ac_mask = ac_mask.reshape(-1, act_dim*3)
        
        state[-1] = state_target[t+1]
        loss[-1] = losses_target[t+1]
        reward[-1] = reward_target[t+1]
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        
        
        
        with torch.no_grad():
            state_error.append( (torch.mean(((state_preds- state_target[t+1,mask])[ac_mask])**2)).detach().cpu().item())
            loss_error.append( torch.mean((loss_preds-losses_target[t+1])**2).detach().cpu().item())
            reward_error.append( torch.mean((reward_preds-reward_target[t+1])**2).detach().cpu().item())
            # print(f'step{t} state_error:{state_error[-1]}, loss_error:{loss_error[-1]}, reward_error:{reward_error[-1]}')

    with open(fold_name, 'a', encoding='utf-8') as f:
        ans = {'state_error':state_error, 'loss_error':loss_error, 'reward_error':reward_error}
        f.write(json.dumps(ans) + '\n')
    print(f'step{t} state_error:{np.mean(state_error)}, loss_error:{np.mean(loss_error)}, reward_error:{np.mean(reward_error)}')
    
    return state_error,loss_error,reward_error



def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        loss_dim,
        model,
        max_ep_len=101,
        device='cuda',
        path_name=None,
    ):

    fold_name = path_name+'/eval_data.txt'

    ans = []
    
    model.eval()
    model.to(device=device)

    model.eval()
    model.to(device=device)

    state_target = torch.from_numpy(env['state']).reshape(max_ep_len ,state_dim).to(device=device, dtype=torch.float32)
    actions = torch.from_numpy(env['action']).reshape(max_ep_len, act_dim).to(device=device, dtype=torch.float32)
    reward_target = torch.from_numpy(env['acc']).reshape(max_ep_len, 1).to(device=device, dtype=torch.float32)
    losses_target = torch.from_numpy(env['loss']).reshape(max_ep_len, loss_dim).to(device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)


    state = torch.zeros((1, state_dim), device=device, dtype=torch.float32)
    state[-1]  = state_target[0,:]
    loss = torch.zeros((1, loss_dim), device=device, dtype=torch.float32)
    loss[-1]  = losses_target[0,:]
    reward = torch.zeros((1, 1), device=device, dtype=torch.float32)
    reward[-1]  = reward_target[0,:]

    mask = torch.ones((state_dim), dtype=torch.bool)
    mask[0::4] = False

    loss_error, state_error, reward_error = [],[],[]

    for t in range(max_ep_len-1):
        # add padding
        action = actions[0:t+1].to(dtype=torch.float32)
        action = torch.cat([action, torch.zeros((1, act_dim), device=device)], dim=0)
        reward = torch.cat([reward, torch.zeros((1,1), device=device)])
        state = torch.cat([state, torch.zeros((1, state_dim), device=device)])
        loss = torch.cat([loss, torch.zeros((1, loss_dim), device=device)])

        state_preds, loss_preds, reward_preds = model.get_states(
            state.to(dtype=torch.float32),
            loss.to(dtype=torch.float32),
            action.to(dtype=torch.float32), 
            reward.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
         
        ac_mask = torch.repeat_interleave(action[-2,], repeats=3,dim=0).bool()
        ac_mask = ac_mask.reshape(-1, act_dim*3)
        
        state[-1,mask] = state_preds
        state[-1,~mask] = state[0,~mask]
        loss[-1] = loss_preds
        reward[-1] = reward_preds
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        
        
        with torch.no_grad():
            state_error.append( (torch.mean(((state_preds- state_target[t+1,mask])[ac_mask])**2)).detach().cpu().item())
            loss_error.append( torch.mean((loss_preds-losses_target[t+1])**2).detach().cpu().item())
            reward_error.append( torch.mean((reward_preds-reward_target[t+1])**2).detach().cpu().item())
            # print(f'step{t} state_error:{state_error[-1]}, loss_error:{loss_error[-1]}, reward_error:{reward_error[-1]}')

    with open(fold_name, 'a', encoding='utf-8') as f:
        ans = {'state_error':state_error, 'loss_error':loss_error, 'reward_error':reward_error}
        f.write(json.dumps(ans) + '\n')
    print(f'step{t} state_error:{np.mean(state_error)}, loss_error:{np.mean(loss_error)}, reward_error:{np.mean(reward_error)}')
    
    return state_error,loss_error,reward_error

# def get_init(data):
#     data_new = np.zeros(like=data)
#     data_new[0,:] = data[0,:]
#     return data_new
    
        