import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
import datetime
import os

os.environ["WANDB_API_KEY"] = 'your wandb api key' 
os.environ["WANDB_MODE"] = "offline"


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def get_val(data):
    # vaild set step = 11, choise vaild data
    # validation_indices = np.arange(0, 660, 11)
    validation_indices = np.arange(0,220, 11)
    validation_data = data[validation_indices]

    train_indices = np.setdiff1d(np.arange(220), validation_indices)
    train_data = data[train_indices]
    return train_data, validation_data

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'+variant['train_type']
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    max_ep_len = 101
    if model_type == 'dt':
        env_targets = [1]  # since BC ignores target, no need for different evaluations

    state_dim = 100*4  
    loss_dim = 100
    act_dim = 100

    # folder name
    save_data_path = './DT_ModelBased/decision_transformer/major_save_data/'

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%b%d-%H_%M')
    fold_name = formatted_datetime + variant['train_type']

    cur_path = save_data_path+fold_name
    os.makedirs(cur_path)


    # load dataset
    state_data = np.load('./DT_ModelBased/training_data/states.npy')  #(sample, round, clients_num, FL_feature)  e.g., (200, 101, 20, 7) (round add initial state)  Only the last four features in this example are valid for state_data.
    loss_data = np.load('./DT_ModelBased/training_data/loss.npy')   #(sample, round, client_loss) e.g., (200, 101, 20)
    accs_data = np.load('./DT_ModelBased/training_data/accs.npy')   #(sample, round, client_loss) e.g., (200, 101, 1)
    action_data = np.load('./DT_ModelBased/training_data/actions.npy') #(sample, round, 1, client_action) e.g., (200, 101, 1, 20)  (action in {0,1} for each client, 1 means selected, 0 means not selected)

    client_num = variant['flnum']
    zeros_data = np.zeros((state_data.shape[0], 1, 1, client_num))
    # zeros_data = np.zeros((660, 1, 1, 20))
    action_data = np.concatenate([action_data,zeros_data], axis=1)
        # trajectories = pickle.load(f)

    state_data, state_data_eva = get_val(state_data)
    loss_data, loss_data_eva = get_val(loss_data)
    accs_data, accs_data_eva = get_val(accs_data)
    action_data, action_data_eva= get_val(action_data)



    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{state_data.shape[0]} trajectories, {state_data.shape[1]} timesteps found')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    num_trajectories = state_data.shape[0]
    num_step = state_data.shape[1]
    pct_traj = variant.get('pct_traj', 1.)


#-----------------------------------------  The above is data processing  ---------------------------------------

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=False,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        s, l, a, r, timesteps, mask = [], [], [], [], [], []
        s = state_data[batch_inds, :,:,3:].reshape(batch_size, -1, state_dim)    # Only the last four features in this example are valid for state_data.
        l = loss_data[batch_inds,:,:].reshape(batch_size, -1, loss_dim)
        a = action_data[batch_inds, :,:].reshape(batch_size, -1, act_dim)
        r = accs_data[batch_inds, :,:].reshape(batch_size, -1, 1)
        tlen = s.shape[1]
        for _ in batch_inds:
            timesteps.append(np.arange(0, s.shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=device)
        l = torch.from_numpy(l).to(dtype=torch.float32, device=device)

        # rtg = r

        return s, a, r, l, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns = []
            # index = np.random.choice(np.arange(num_eval_episodes),size=batch_size,replace=False)
            index = np.arange(0,20,2)
            for i in index:
                env = {'state':state_data_eva[i,:,:,3:],'loss':loss_data_eva[i],'acc':accs_data_eva[i],'action':action_data_eva[i]}
                with torch.no_grad():
                    if model_type == 'dt':
                        state_error, loss_error,reward_error= evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            loss_dim,
                            model,
                            max_ep_len=max_ep_len,
                            device=device,
                            path_name = cur_path
                        )
                returns.append([state_error,loss_error,reward_error])
            return {
                f'target_{target_rew}_return_mean': np.mean(returns,axis=0),
                f'target_{target_rew}_return_std': np.std(returns,axis=0),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            loss_dim=loss_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            # loss_fn=lambda s_hat, l_hat, r_hat, s, l, r, a: torch.sum((s_hat - s)**2)/(torch.sum(a)*3) + torch.mean((l_hat-l)**2) + torch.mean((r_hat-r)**2),
            # loss_fn=lambda s_hat, l_hat, r_hat, s, l, r, a: torch.mean((s_hat-s)**2) + torch.mean((s_hat[a]-s[a])**2) + torch.mean((l_hat-l)**2) + torch.mean((r_hat-r)**2),
            num_steps=variant['num_steps_per_iter'],
            loss_fn=lambda s_hat, l_hat, r_hat, s, l, r, a: torch.mean((s_hat[a]-s[a])**2) + torch.mean((l_hat-l)**2) + torch.mean((r_hat-r)**2),

            eval_fns=[eval_episodes(tar) for tar in env_targets],
            path = cur_path,
            train_type = variant['train_type']
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration( iter_num=iter, print_logs=True)
        if iter%10 == 9 or iter == variant['max_iters']-1:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, cur_path+f'/checkpoint_{iter}.pth')

        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flnum', type=int, default=20)
    parser.add_argument('--env', type=str, default='FL20')
    parser.add_argument('--dataset', type=str, default='MNIST')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=101)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--train_type', type=str, default='teacher')
    parser.add_argument('--samp_prob', type=str, default='sigmoid')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    
    args = parser.parse_args()

    experiment('model-based-experiment', variant=vars(args))
