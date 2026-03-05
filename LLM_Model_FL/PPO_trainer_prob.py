#RL trainer.py
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from trl.core import clip_by_value, logprobs_from_logits, masked_mean, masked_whiten
import torch.nn.functional as F
from accelerate import Accelerator
import random, logging, math, warnings, copy

import numpy as np
import os
# from env_reward import FL_reward
from transformers import Trainer,get_scheduler, PreTrainedModel, GenerationConfig
from peft import get_peft_model,get_peft_model_state_dict, PeftModel
from trl import AutoModelForCausalLMWithValueHead
import time

WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.getLogger(__name__)
        

class PPOModel(nn.Module):
    def __init__(self, actor_model, critic_model):
        super().__init__()
        self.actor_model = actor_model 
        self.critic_model = critic_model 
    
    def forward(self, sequences):
        actor_logits = self.actor_model(**sequences, return_dict=True).logits
        critic_values = self.critic_model(**sequences)[-1][:,-1:]


        return actor_logits, critic_values

class PPOTrainer:
    def __init__(self, args, ppo_model, reward_env):
        self.args = args
        # self.device = self.args.device_ppo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ppo_model = ppo_model
        self.actor_model = ppo_model.actor  # Current Policy Model (with LoRA)
        self.reward_model = reward_env  # reward model
        self.critic_model = ppo_model.critic
        # self.ref_model = ppo_model.ref_model  # ref model
        self.tokenizer = self.ppo_model.tokenizer 

        self.model = PPOModel(self.actor_model, self.critic_model)
        
        self.optimizer = torch.optim.AdamW(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()),
            lr=args.learning_rate_ppo,
            # eps=args.adam_epsilon
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision='fp16' if self.args.fp16 else None,
            log_with=self.args.report_to, project_dir="./run_log",
            device_placement=False
            
        )
        self.accelerator.init_trackers(
            project_name=self.args.project_name,
            config=vars(self.args) 
        )

        # create optimizer and scheduler 
        self.lr_scheduler = self.create_scheduler(self.optimizer, max_update_steps=self.args.ppo_epochs * self.args.max_iteration)

        # Prepare models and optimizers with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model,
            self.optimizer)

        self.model, self.optimizer= self.accelerator.prepare(self.model, self.optimizer)

    def create_scheduler(self, optimizer, max_update_steps):
        lr_scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=self.args.warmup_steps*max_update_steps, num_training_steps=max_update_steps)
        return lr_scheduler
   
    @torch.no_grad()
    def get_reward(self, data_states, state_model, loss_model, actions, acc_model, timesteps, round): 
        acc, time_c, energy, data_states, state, loss, acc_reward, next_prompt = self.reward_model.get_reward_state(data_states, state_model, loss_model, actions, acc_model, timesteps, round)
        return  acc, time_c, energy, data_states, state, loss, acc_reward, next_prompt

    @torch.no_grad()
    def get_model_output(self, sequences):
        
        actor_logits, critic_values= self.model(sequences)  
        # ref_logits = self.ref_model(**sequences, return_dict=True).logits
        
        return actor_logits, critic_values#, ref_logits
    
    def get_log_probs(self, logits, labels):
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[..., self.ppo_model.allowed_token_ids] = True
        masked_logits = logits.masked_fill(~mask, -torch.inf)
        output_num = -self.args.sele_clients*2
        
        output_token = labels[:,output_num:]
        for i in range(output_num+1,0):
            if i%2 == 0:
                mask_token = output_token[:,0:i-output_num:2]
                masked_logits[:,i,mask_token] = -torch.inf
            else:
                masked_logits[:,i,:] =  -torch.inf
                masked_logits[:,i,220] = 1
        
        log_probs = F.log_softmax(masked_logits, dim=-1) 
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        log_probs_labels[log_probs_labels==-torch.inf] = -1e10
        return log_probs_labels.squeeze(-1)

    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps) 

    def masked_var(self, data, mask, dim=None):
        mean = self.masked_mean(data, mask, dim=dim)
        centered_values = data - mean
        var = self.masked_mean(centered_values**2, mask, dim=dim)
        return var
    
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Recursively unwraps a model from potential containers (as used in distributed training).

        Args:
            model (`torch.nn.Module`): The model to unwrap.
        """
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return self.unwrap_model(model.module)
        else:
            return model

    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy 
    
    def get_responses_mask(self, sequences_mask, prompts_without_padding):
        batch_size = sequences_mask.shape[0]
        responses_mask = []
        for i in range(batch_size):
            prompt = prompts_without_padding[i]
            response_mask = torch.zeros_like(sequences_mask[i])
            response_mask[len(prompt):] = sequences_mask[i][len(prompt):]
            responses_mask.append(response_mask)
        return torch.stack(responses_mask)

    
    def get_advantages_and_returns(self, values, rewards):

        # masks = responses_mask[:, 1:] 
        
        lastgaelam = 0 
        advantages_reversed = []
        length = rewards.size()[-1]

        for t in reversed(range(length)):
            nextvalues = values[t + 1] if t < length - 1 else 0.0
            delta = rewards[t] + self.args.gamma * nextvalues - values[ t]  
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam        
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1])
        returns = advantages + values     
        
        # if self.args.use_advantage_norm:
        #     advantages = self.masked_whiten(advantages, masks)
            
        return advantages.detach(), returns


    # def compute_rewards_with_kl_penalty(self, rewards_score, actor_log_probs, ref_log_probs, responses_mask):
    #     masks = responses_mask[:, 1:]
        
    #     batch_size = rewards_score.shape[0]
    #     # batch_size = 1
    #     rewards_with_kl_penalty, kl_penalty_all = [], []
    #     for i in range(batch_size):
    #         mask = masks[i]
            
    #         kl = actor_log_probs[i] - ref_log_probs[i]
    #         if self.args.kl_penalty_method == 'abs':
    #             kl = torch.abs(kl)
    #         elif self.args.kl_penalty_method == 'mse':
    #             kl = kl ** 2 * 0.5 
                
    #         kl_penalty = - self.args.kl_penalty_beta * kl 
    #         kl_penalty_all.append(kl_penalty)

    #         # if self.args.reward_score_clip is not None:
    #         #     rewards_score[i] = torch.clamp(rewards_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)
            
    #         end_index = mask.nonzero()[-1].detach().item()
    #         kl_penalty[end_index] += rewards_score[i]

    #         rewards_with_kl_penalty.append(kl_penalty)
    #     return torch.stack(rewards_with_kl_penalty), torch.stack(kl_penalty_all)

    def actor_loss(self, actor_log_probs, mini_batch_actor_log_probs, advantages, mask):
        # resp_len = self.args.sele_clients
        # resp_mini = mini_batch_actor_log_probs[:,-sele_clients]
        ratio = torch.exp(mini_batch_actor_log_probs - actor_log_probs)* mask
        ratio[ratio==0] = 1
        ratio = torch.prod(ratio)
        loss1 = -advantages * ratio
        loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.ratio_clip,
                                             1.0 + self.args.ratio_clip)

        # loss = self.masked_mean(torch.max(loss1, loss2), mask)
        loss = torch.max(loss1, loss2)
        return loss, ratio  


    def critic_loss(self, critic_values, mini_batch_critic_values, returns, mask):
        
        critic_values_clip = torch.clamp(
            mini_batch_critic_values,
            critic_values - self.args.value_clip,
            critic_values + self.args.value_clip,
        )
        values_error = (mini_batch_critic_values - returns)**2 
        values_clip_error = (critic_values_clip - returns)**2 
         # loss = 0.5 * self.masked_mean(torch.max(values_error, values_clip_error), mask)
        loss = 0.5 *torch.max(values_error, values_clip_error)
        
        return loss, values_error 

    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict 


    def save_checkpoint(self, model, output_dir, step, adapter_name="default", state_dict=None):

        if self.unwrap_model(model) is not model:
            model = self.unwrap_model(model)
            
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            if hasattr(model, "v_head"):
                state_dict = self.get_state_dict(model)
            else:
                state_dict = model.state_dict()

        if isinstance(model, PreTrainedModel):  
            model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if hasattr(model, "peft_config"):
                adapter_state_dict = get_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                adapter_state_dict = get_peft_model_state_dict(model.pretrained_model, state_dict, adapter_name=adapter_name)

            if hasattr(model, "v_head"):
                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v 
            torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
        try:
            if hasattr(model, "peft_config"):
                model.peft_config.save_pretrained(output_dir)
            elif isinstance(model, AutoModelForCausalLMWithValueHead):
                model.pretrained_model.peft_config.save_pretrained(output_dir)

        except AttributeError:
            if hasattr(model, "peft_config"):
                model.peft_config[adapter_name].save_pretrained(output_dir)
            else:
                model.pretrained_model.peft_config[adapter_name].save_pretrained(output_dir)


        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def record_logs(self, batch):

        mask = batch["responses_mask"][:, 1:]
        prompt_lens = torch.tensor([len(prompt) for prompt in batch["prompts_ids"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in batch["responses_ids"]], dtype=torch.float)

        logs = dict()
        ## params
        logs["lr"] = self.optimizer.param_groups[0]['lr']
        
        ## loss
        logs["loss/actor"] = batch["actor_loss"]
        logs["loss/entropy"] = batch["entropy"]
        logs["loss/critic"] = batch["critic_loss"]
        # logs["loss/extra"] = batch["extra_loss"]

        logs["exp_data/reward_score_mean"] = torch.mean(batch["rewards_score"])
        logs["exp_data/energy"] = torch.mean(batch["energy"])
        # logs["exp_data/reward_score_var"] = torch.var(batch["rewards_score"]) 
        
        # logs["exp_data/kl_penalty_mean"] = self.masked_mean(batch["kl_penalty"], mask)
        # logs["exp_data/kl_penalty_var"] = self.masked_var(batch["kl_penalty"], mask)

        # logs["exp_data/rewards_with_kl_penalty_mean"] = self.masked_mean(batch["rewards_with_kl_penalty"], mask)
        # logs["exp_data/rewards_with_kl_penalty_var"] = self.masked_var(batch["rewards_with_kl_penalty"], mask)
        
        logs["exp_data/actor_perplexity"] = math.exp(torch.mean(batch["actor_ce_loss"]))
        # logs["exp_data/ref_perplexity"] = math.exp(torch.mean(batch["ref_ce_loss"]))
        
        ## actor
        logs["actor/advantages_mean"] = self.masked_mean(batch["advantages"], mask)
        # logs["actor/advantages_var"] = self.masked_var(batch["advantages"], mask)
        
        logs["actor/ratio_mean"] = self.masked_mean(batch["ratio"], mask)
        # logs["actor/ratio_var"] = self.masked_var(batch["ratio"], mask)
        
        ## critic
        logs["critic/returns_mean"] = self.masked_mean(batch["returns"], mask)
        # logs["critic/returns_var"] = self.masked_var(batch["returns"], mask)

        logs["critic/values_error_mean"] = self.masked_mean(batch["values_error"], mask)
        # logs["critic/values_error_var"] = self.masked_var(batch["values_error"], mask)
        
        ## length
        logs["length/prompts_length_mean"] = torch.mean(prompt_lens)
        # logs["length/prompts_length_var"] = torch.var(prompt_lens)
        
        logs["length/responses_length_mean"] = torch.mean(response_lens)
        # logs["length/responses_length_var"] = torch.var(response_lens)
        
        return logs

    def process_sequences(self, prompts_ids, responses_ids):
        # seq: [0 0 0 0, prompt, response, 0 0 0 0] change to [prompt, response, 0 0 0 0]
        
        prompts_without_padding, responses_without_padding = [], []
        batch_size = prompts_ids.shape[0]
        for i in range(batch_size):
            response = responses_ids[i]
            prompt = prompts_ids[i] 
            prompt_left_padding_length = (prompt == self.tokenizer.pad_token_id).sum().item()
            response_length = (response != self.tokenizer.pad_token_id).sum().item()
            prompt_without_padding = prompt[prompt_left_padding_length:]
            response_without_padding = response[:response_length]
            
            prompts_without_padding.append(prompt_without_padding.to(self.device))
            responses_without_padding.append(response_without_padding.to(self.device))
        
        
        new_sequences = [torch.cat([q, r]) for q, r in zip(prompts_without_padding, responses_without_padding)]
        sequences = torch.nn.utils.rnn.pad_sequence(
            new_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        sequences = dict(
            input_ids=sequences.to(self.device),
            attention_mask=sequences.ne(self.tokenizer.pad_token_id).long().to(self.device)
        )
        
        return torch.stack(prompts_without_padding), torch.stack(responses_without_padding), sequences
    
    def get_experience_data(self, data_states, prompts_ids, state_model, loss_model, actions, acc_model, timesteps, round, ref_action=None,temperature=None):
        if not ref_action:
            responses_ids = self.ppo_model.generate_act(prompts_ids, temperature)
            sentence = self.tokenizer.convert_ids_to_tokens( responses_ids[0], skip_special_tokens=True)
            # action = list(map(int, sentence))
            action = [int(c) for c in sentence if c.isdigit()]
        else:
            sentence = ref_action
            sentence_ids = self.tokenizer.encode(sentence, return_tensors="pt")
            sentence_ids = torch.cat([sentence_ids, torch.tensor([[self.tokenizer.eos_token_id]])], dim=-1).to(self.device)
            responses_ids = torch.cat([prompts_ids, sentence_ids], dim=-1)

        
        prompts_without_padding, responses_without_padding, sequences = self.process_sequences(prompts_ids, responses_ids) 
        
        
        actor_logits, critic_values = self.get_model_output(sequences)


        temp = torch.zeros((1,20)).to(self.device)
        for i in action:
            temp[0][i] = 1
        if actions == None:
            actions = temp
        else:
            actions = torch.cat([actions, temp], dim=0)
        actions = actions.to(self.device, dtype=torch.float32)

        acc, time_c, energy, data_states, state, loss, acc_model, next_prompt = self.get_reward(data_states, state_model, loss_model, actions, acc_model, timesteps, round)
        done = 0

        responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)
        actor_log_probs = self.get_log_probs(actor_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        actor_ce_loss = -self.masked_mean(actor_log_probs, responses_mask[:, 1:], dim=-1)

        # ref_log_probs = self.get_log_probs(ref_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        # ref_ce_loss = -self.masked_mean(ref_log_probs, responses_mask[:, 1:], dim=-1)

        # responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)
        
        # rewards_with_kl_penalty, kl_penalty, rewards_score = self.compute_rewards_with_kl_penalty(ref_values, actor_log_probs, ref_log_probs, responses_mask)

        # critic_values = critic_values[:, :-1] * responses_mask[:, 1:] 
        # rewards_with_kl_penalty = rewards_with_kl_penalty * responses_mask[:, 1:]  
        # advantages, returns = self.get_advantages_and_returns(critic_values, rewards_with_kl_penalty, responses_mask)
        # advantages, returns = None, None

        return dict(
            prompts_ids=prompts_without_padding,
            responses_ids=responses_without_padding,
            responses_mask=responses_mask,
            sequences_ids=sequences["input_ids"],
            sequences_mask=sequences["attention_mask"],
            actor_log_probs=actor_log_probs,
            # ref_log_probs=ref_log_probs,
            # rewards_with_kl_penalty=None,
            # rewards_score=torch.tensor([ref_values]),
            time=torch.tensor([time_c]),
            acc=torch.tensor([acc]),
            energy=torch.tensor([energy]),
            # kl_penalty=None,
            critic_values=critic_values,
            # advantages=None,
            # returns=None,
            actor_ce_loss=actor_ce_loss,
            # ref_ce_loss=ref_ce_loss,
        ), data_states, state, loss, acc_model, actions, done, next_prompt


    def get_mini_dataset(self, data_buffer):

        mini_dataset = []
        batch_size = data_buffer[0]["exp"]["sequences_ids"].shape[0]
        for item in data_buffer:
            experience_data, batch_extra_data = item['exp'], item['extra']
            index = 0 
            while index < batch_size:
                dic = {}
                for k, v in experience_data.items():
                    if k in ["prompts_ids", "responses_ids"]:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size]
                    else:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                        
                if batch_extra_data is not None:
                    for k, v in batch_extra_data.items():
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                
                mini_dataset.append(dic)
                index += self.args.per_device_mini_train_batch_size
 
        return mini_dataset
    

    def print_logs(self, all_logs, update_steps):

        all_logs_merged = {}
        for key in all_logs[0]:
            all_logs_merged[key] = torch.mean(torch.tensor([log[key] for log in all_logs])).to(self.device)
        
        all_logs_merged["exp_data/reward_score_var"] = torch.var(torch.tensor([log["exp_data/reward_score_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["length/prompts_length_var"] = torch.var(torch.tensor([log["length/prompts_length_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["length/responses_length_var"] = torch.var(torch.tensor([log["length/responses_length_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["actor/advantages_var"] = torch.var(torch.tensor([log["actor/advantages_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["actor/ratio_var"] = torch.var(torch.tensor([log["actor/ratio_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["critic/returns_var"] = torch.var(torch.tensor([log["critic/returns_mean"] for log in all_logs])).to(self.device)
        all_logs_merged["critic/values_error_var"] = torch.var(torch.tensor([log["critic/values_error_mean"] for log in all_logs])).to(self.device)
        
        
        if self.accelerator.is_main_process:
            logs = {}
            for k, v in all_logs_merged.items():
                logs[k] = v.cpu().numpy().item()
            self.accelerator.log(logs, step=int(update_steps))

            if update_steps > 0 and update_steps % self.args.logging_steps == 0:
                actor_loss, critic_loss = logs["loss/actor"], logs["loss/critic"]
                rewards_score = logs["exp_data/reward_score_mean"]
                energy = logs["exp_data/energy"]
                lr = logs["lr"]
                print(f'update_steps:{update_steps}|lr:{lr}|actor_loss:{actor_loss}, critic_loss:{critic_loss}, rewards_mean:{rewards_score}, energy:{energy}')
    


    def train_step(self, batch_mini_data, step):
        
        responses_mask = batch_mini_data["responses_mask"]
        sequences = {"input_ids": batch_mini_data["sequences_ids"], "attention_mask": batch_mini_data["sequences_mask"]}

        with self.accelerator.accumulate(self.model):
            mini_batch_actor_logits, mini_batch_critic_values = self.model(sequences)
        
                
            mini_batch_actor_log_probs = self.get_log_probs(mini_batch_actor_logits[:, :-1, :], batch_mini_data["sequences_ids"][:, 1:]) 
            entropy = self.get_entropy(mini_batch_actor_logits[:, :-1, :], responses_mask[:, 1:])            
            actor_loss, ratio = self.actor_loss(batch_mini_data["actor_log_probs"], mini_batch_actor_log_probs, batch_mini_data["advantages"], responses_mask[:, 1:])             
            
            critic_loss, values_error = self.critic_loss(batch_mini_data["critic_values"], mini_batch_critic_values, batch_mini_data["returns"], responses_mask[:, 1:])
            

            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss
            # actor_loss_p = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy 
            # cirtic_loss_p = self.args.critic_loss_weight * critic_loss

            
            self.accelerator.backward(loss)
            
            if self.args.max_grad_norm is not None:
                params = [p for n, p in self.actor_model.named_parameters() if p.requires_grad] + [p for n, p in self.critic_model.named_parameters() if p.requires_grad]
                    
                torch.nn.utils.clip_grad_norm_(
                    parameters=params,
                    max_norm=self.args.max_grad_norm
                )
                # self.accelerator.clip_grad_norm_(
                #     parameters=params,
                #     max_norm=self.args.max_grad_norm
                # )

            self.optimizer.step()
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()

            self.optimizer.zero_grad()

        return dict(
            all_loss=loss.detach(),
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            entropy=entropy.detach(),
            ratio=ratio.detach(),
            values_error=values_error.detach(),
            
        )
    
    def traj_process(self,sigle_traj,acc_init): # 单个轨迹处理
        length = len(sigle_traj)
        reward_reversed =[]
        acc_delate = [0.0 for i in range(len(sigle_traj))]
        for t in range(len(sigle_traj)):
            # sigle_traj[t]['exp']['energy'] += sigle_traj[t-1]['exp']['energy'] if t > 0 else 0.0
            acc_delate[t] = sigle_traj[t]['exp']['acc'] - (sigle_traj[t-1]['exp']['acc'] if t > 0 else acc_init.cpu().numpy().item())
            sigle_traj[t]['exp']['rewards_score'] = self.reward_model.cal_reward(sigle_traj[t]['exp']['acc'],  
                                                                                 sigle_traj[t]['exp']['time'], 
                                                                                 sigle_traj[t]['exp']['energy'], acc_delate[t]).to(self.device)
            # sigle_traj[t]['exp']['rewards_score'] = self.reward_model.cal_reward2(sigle_traj[t]['exp']['acc'],  sigle_traj[t]['exp']['time'], sigle_traj[t]['exp']['energy'])

        # for t in reversed(range(len(sigle_traj))):
        #     nextvalues = sigle_traj[t+1]['exp']['rewards_score'] if t < length - 1 else 0.0
        #     delta = sigle_traj[t]['exp']['rewards_score'] + self.args.gamma * nextvalues      
        #     reward_reversed.append(delta)
        # rewards_score = reward_reversed[::-1]
        # for i, item in enumerate(sigle_traj):
        #     item['exp']['rewards_score'] = rewards_score[i]

            # sigle_traj = { key: torch.cat([step['exp'][key] for step in sigle_traj],dim=0)
            #     for key in sigle_traj[0]['exp'].keys()
            # } if sigle_traj else {}
        return sigle_traj

    def sigle_advantage(self, sigle_traj):
        rewards = torch.stack([x['exp']['rewards_score'][0] for x in sigle_traj])
        values = torch.stack([x['exp']['critic_values'][0][-1] for x in sigle_traj])
        # masks = torch.stack([x['exp']['responses_mask'] for x in sigle_traj])
        advantages, returns = self.get_advantages_and_returns(values,rewards)
            
        for i, temp in enumerate(sigle_traj):
            temp['exp']['advantages'] = advantages[i].unsqueeze(dim=0)
            temp['exp']['returns'] = returns[i].unsqueeze(dim=0)
        return sigle_traj
    
    def process_buffer(self, data_buffer):
        advantages = []
        for item in data_buffer:
            experience_data, batch_extra_data = item['exp'], item['extra']
            advantages.append(experience_data['advantages'])
        # Normalizing the advantages
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for advantage, item in zip(advantages, data_buffer):
            item['exp']['advantages'] = torch.tensor([advantage])
            # item['exp']['kl_penalty'] = self.compute_kl_penalty(item['exp']["actor_log_probs"], item['exp']['ref_log_probs'], item['exp']['responses_mask'])


    def train(self):
        # if self.is_world_process_zero():
        # Train!
        max_iteration = self.args.max_iteration       
        max_step = self.args.ppo_epochs * max_iteration * 100 * self.args.traj_batch_llm
        logger.info("***** Running training *****")
        logger.info(f"  Num iteration = { max_iteration:,}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total steps = {max_step}")

        step = 0 
        collect_step = 0
        data_buffer = list()
        all_logs = list()
        done_num = 0  
        

        progress_bar = tqdm(total=max_step )
        
        data_states_init = []
        # if step < 50000:
        #     self.reward_model.env.reset(0, 42)
        #     temperature = 1.0
        # elif step < 120000:
        #     self.reward_model.env.reset(0, random.randint(0, 2**10))
        #     temperature = 0.8
        # else:
        #     self.reward_model.env.reset(random.randint(0, 999), random.randint(0, 2**10))
        #     temperature = 0.6
        self.reward_model.env.reset(0, 42)
        temperature = 1.0
        # 初始状态
        global_parameters = {}
        for key, var in self.reward_model.env.net.state_dict().items():
            global_parameters[key] = var.clone()
        for i in range(self.reward_model.env.args['num_of_clients']):
            data_states_init.append(self.reward_model.env.init_static_state(i))
        for i in range(self.reward_model.env.args['num_of_clients']):    
            self.reward_model.env.update_state_broadcast(data_states_init[i], i, global_parameters)
        acc_init = self.reward_model.env.valid( global_parameters,self.reward_model.env.testDataLoader)
        state_model_init, loss_model_init, acc_model_init = self.reward_model.env.get_state_model(data_states_init, acc_init)
        state_model_init =  torch.tensor(state_model_init).reshape(1, self.args.state_dim).to(device=self.device,dtype=torch.float32)
        loss_model_init =  torch.tensor(loss_model_init).reshape(1, self.args.loss_dim).to(device=self.device,dtype=torch.float32)
        acc_model_init =  acc_model_init.reshape(1, 1).to(device=self.device,dtype=torch.float32)
        actions_init = None

        last_inf = f'The initial global model accuracy is {acc_init}'
        round = 1
        ref = '[1 2 3 4]'
        prompts_init = self.reward_model.env.process2txt(last_inf, round, data_states_init, self.reward_model.env.args['num_of_clients'], self.reward_model.env.num_in_comm, self.reward_model.env.args['time_lim'], ref)
        prompts_ids_init = self.tokenizer.encode(prompts_init, return_tensors="pt").to(self.device) 

        iter = 0
        update_steps = 0
        while iter <= max_iteration:
            self.reward_model.env.seed_everything(int(time.time()))
            sigle_traj = []

            #LLM  Interaction Action Trajectory       
            for traj_num in range(self.args.traj_batch_llm):
                current_ep_reward = 0
                sigle_traj = []
                data_states = copy.deepcopy(data_states_init)            
                state_model, loss_model, acc_model, actions = copy.deepcopy(state_model_init), copy.deepcopy(loss_model_init), copy.deepcopy(acc_model_init), copy.deepcopy(actions_init)
                timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
                prompts_ids = prompts_ids_init         
                timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
                for t in range(1, self.args.max_ep_len+1):
                    experience_data, data_states, state_model, loss_model, acc_model, actions, done, next_prompt = self.get_experience_data(data_states,prompts_ids, state_model, loss_model, actions, acc_model, timesteps, t, temperature=temperature)
                    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (round)], dim=1)
                    sigle_traj.append({'exp': experience_data, 'extra': None})
                    collect_step +=1 
                    if done:
                        done_num += 1
                        break
                    if t == self.args.max_ep_len:
                        done_num += 1
                    prompts_ids = self.tokenizer.encode(next_prompt, return_tensors="pt").to(self.device)

                sigle_traj = self.traj_process(sigle_traj, acc_init)
   
                temp = self.sigle_advantage(sigle_traj)
                data_buffer.extend(temp)


            if collect_step >= self.args.mini_data_buffer_nums: 
                self.process_buffer(data_buffer)
                done_num = 0 
                mini_dataset = self.get_mini_dataset(data_buffer)
                
                log_reward = dict()
                log_reward["reward_traj"] = torch.mean(torch.tensor([log["rewards_score"] for log in mini_dataset])).cpu().numpy().item()
                self.accelerator.log(log_reward, step=int(iter))
                print(f"Iteration{' '}{iter} reward_mean: {log_reward['reward_traj']}")
                random.shuffle(mini_dataset) 
                data_buffer.clear()
                for ppo_epoch in range(self.args.ppo_epochs): #buffer
                    update_steps += 1
                    for j, batch_mini_data in enumerate(mini_dataset): 
                        step += self.args.per_device_mini_train_batch_size 
                        result = self.train_step(batch_mini_data, step)
                        batch_mini_data.update(result)

                        progress_bar.update(self.args.per_device_mini_train_batch_size )

                        logs = self.record_logs(batch_mini_data)
                        all_logs.append(logs)
                        
                        # update_steps = step / self.args.gradient_accumulation_steps / self.args.per_device_mini_train_batch_size

                        # if step > 0 and step % (self.args.gradient_accumulation_steps * self.args.per_device_mini_train_batch_size) == 0:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.print_logs(all_logs, update_steps) 
                    all_logs.clear()
                        
                    if update_steps > 0 and (update_steps % self.args.save_steps) == 0:
                        
                        # if self.is_world_process_zero():
                        self.save_checkpoint(self.actor_model, self.args.output_dir, int(update_steps))
                        self.save_checkpoint(self.critic_model, self.args.critic_output_dir, int(update_steps))

                    random.shuffle(mini_dataset) 
                    torch.cuda.empty_cache()
                collect_step = 0
                iter += 1

        progress_bar.close()
        self.accelerator.end_training()

