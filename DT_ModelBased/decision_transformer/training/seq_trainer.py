import numpy as np
import torch
import json
from decision_transformer.training.trainer import Trainer
import copy


class SequenceTrainer(Trainer):

    def train_step(self, iter_num):
        states, actions, rewards, loss_state,  timesteps, attention_mask = self.get_batch(self.batch_size)
        
        states_target = torch.clone(states)[:,1:]
        states_target = states_target.reshape(states.shape[0],states.shape[1]-1,100,4)
        states_target = states_target[:,:,:,1:]
        states_target = states_target.reshape(states.shape[0],states.shape[1]-1,-1)

        loss_target = torch.clone(loss_state)[:,1:]
        acc_target = torch.clone(rewards)[:,1:]

        mask = torch.repeat_interleave(actions, repeats=3,dim=2).bool()

        if self.train_type == 'schedul':
            states, rewards, loss_state = self.scheduled_learning(iter_num, states, loss_state, actions,  rewards, timesteps, attention_mask)

        state_preds, loss_preds, action_preds, reward_preds = self.model.forward(
            states, loss_state, actions,  rewards, timesteps, attention_mask=attention_mask,
        )

        states_dim = state_preds.shape[2]
        loss_dim = loss_preds.shape[2]
        reward_dim = reward_preds.shape[2]
        action_dim = actions.shape[2]

        state_preds = state_preds[:,:-1].reshape(-1, states_dim)[attention_mask[:,:-1].reshape(-1) > 0]
        loss_preds = loss_preds[:,:-1].reshape(-1, loss_dim)[attention_mask[:,:-1].reshape(-1) > 0]
        reward_preds = reward_preds[:,:-1].reshape(-1, reward_dim)[attention_mask[:,:-1].reshape(-1) > 0]

        states_target = states_target.reshape(-1, states_dim)[attention_mask[:,:-1].reshape(-1) > 0]
        loss_target = loss_target.reshape(-1, loss_dim)[attention_mask[:,:-1].reshape(-1) > 0]
        acc_target = acc_target.reshape(-1, reward_dim)[attention_mask[:,:-1].reshape(-1) > 0]

        mask = mask[:,:-1].reshape(-1, action_dim*3)[attention_mask[:,:-1].reshape(-1) > 0]
        loss = self.loss_fn(
            state_preds, loss_preds, reward_preds,
            states_target, loss_target, acc_target, mask
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/state_error'] = (torch.mean((state_preds[mask]-states_target[mask])**2)).detach().cpu().item()
            self.diagnostics['training/loss_error'] = torch.mean((loss_preds-loss_target)**2).detach().cpu().item()
            self.diagnostics['training/reward_error'] = torch.mean((reward_preds-acc_target)**2).detach().cpu().item()

        # print(f'step:{iter_num} ',self.diagnostics)

        fold = self.path+'/train_data.txt'
        with open(fold, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.diagnostics)+ '\n')
        
        return loss.detach().cpu().item()


    def scheduled_learning(self, iter_num, states, loss_state, actions,  rewards, timesteps, attention_mask):
        mask = torch.ones((states.shape[-1]), dtype=torch.bool)
        mask[0::4] = False
        with torch.no_grad():
            state_preds, loss_preds, action_preds, reward_preds = self.model.forward(
            states, loss_state, actions,  rewards, timesteps, attention_mask=attention_mask,
        )

        state_whole = copy.deepcopy(states)
        state_whole[:,:,mask] = state_preds
        sampling_prob = self.get_scheduled_sampling_prob(step=iter_num)
        for batch in range(states.shape[0]):
            for step in range(1, states.shape[1]):
                use_teacher_forcing = (torch.rand(1) <= sampling_prob)
                if not use_teacher_forcing:
                    states[batch, step, :] = state_whole[batch, step-1, :]
                    loss_state[batch, step, :] = loss_preds[batch, step-1, :]
                    rewards[batch, step, :] = reward_preds[batch, step-1, :]
        return states, rewards, loss_state




    def get_scheduled_sampling_prob(self, step, mode='linear'):
        """
        return Scheduled Sampling probability
        - real token first, then generating token
        - mode: 'linear' 或 'sigmoid' Control Probability Change Curve
        """
        if mode == 'linear':
            return max(0.01, 1.0 - step / 100)  
        elif mode == 'sigmoid':
            k = 1
            return 1 / (1 + np.exp(k*(step - 20)))
        elif mode == 'exp':
            k = 0.5
            return  np.exp(-k*step)
        else:
            return 1.0  