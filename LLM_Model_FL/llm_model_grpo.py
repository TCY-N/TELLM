import torch 
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from trl import AutoModelForCausalLMWithValueHead
from torch.optim import AdamW
from peft.tuners.lora import LoraLayer 
import os
import re
from transformers import LogitsProcessor
    
class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids, prompt_length, space_token_id, maxlenght):
        """
        - allowed_token_ids: Set of generated content tokens
        - prompt_length
        - space_token_id
        """
        self.allowed_token_ids = set(allowed_token_ids)
        self.prompt_length = prompt_length
        self.space_token_id = space_token_id
        self.used_tokens = set()  
        self.maxlength = maxlenght

    def __call__(self, input_ids, scores):
        current_length = input_ids.shape[-1]
        generated_length = current_length - self.prompt_length
        
        if generated_length >= self.maxlength:
            return scores
        
        generated_tokens = input_ids[0, self.prompt_length:].tolist()
        
        if generated_length % 2 == 0:
            valid_token_ids = [
                tid for tid in self.allowed_token_ids 
                if tid not in self.used_tokens
            ]
            
            mask = torch.full_like(scores, float('-inf'))
            if valid_token_ids:
                mask[:, list(valid_token_ids)] = 0
            else:  
                return scores + mask
        else:
            mask = torch.full_like(scores, float('-inf'))
            mask[:, self.space_token_id] = 0
        
        scores = scores + mask
        
        if generated_length > 0 and (generated_length-1) % 2 == 0:
            prev_token = generated_tokens[-1]
            if prev_token in self.allowed_token_ids:
                self.used_tokens.add(prev_token)
        
        return scores


class LLaMAModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
        self.lora_config = self._apply_lora()
        self.actor = self._init_actor()
        # self.critic = self._init_critic()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        if config.llm_model == 'llama':
            self.tokenizer.pad_token = "<|reserved_special_token_9|>"
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        # self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_path)
        # self.ref_model.to(self.config.device_ppo)
        self.allowed_token_ids = self._limit_token()
        # if self.config.llm_model == 'gpt':
        #     self.space_token = self.tokenizer.convert_tokens_to_ids(' ')
        # else:
        self.space_token = 220
        

    def _apply_lora(self):
        """Applying LoRA to the model"""
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            # target_modules=["c_attn"],  # Modules requiring LoRA application
            target_modules=["c_attn", "c_proj"] if self.config.llm_model == 'gpt' else \
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,  
            bias="none"  
        )
        return lora_config 
        
    def _init_actor(self):
        model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
        if self.config.actor_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.config.actor_peft_path, is_trainable=True)
        else:
            model = get_peft_model(model, self.lora_config)
        model.to(self.device)       
        return model


    def _limit_token(self):
        if self.config.llm_model == 'qwen':
            allowed_tokens = [chr(i+65) for i in range(int(self.config.num_of_clients))]
        else:
            allowed_tokens = [str(i) for i in range(int(self.config.num_of_clients))]  
        allowed_token_ids = self.tokenizer.convert_tokens_to_ids(allowed_tokens)
        return allowed_token_ids 
    
    def logits_processor(self, input_ids, scores):
        mask = torch.ones_like(scores) * -float("inf")
        mask[:, self.allowed_token_ids] = 0  
        return scores + mask

    def generate_act(self, prompts_ids, temperature):
        """Generate text (for RL sampling)"""
        # prompts_ids = self.tokenizer.encode(inputs, return_tensors="pt")
        prompt_length = prompts_ids.shape[1]
        maxlenght = self.config.sele_clients * 2
        processor = ConstrainedLogitsProcessor(self.allowed_token_ids, prompt_length, self.space_token, maxlenght) 
        
        self.actor.eval()
        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": maxlenght,
            "min_new_tokens": maxlenght, 
            "_from_model_config": False,
            "logits_processor":[processor],
            "temperature":temperature, 
            "do_sample":True
        }
        sequences_ids = self.actor.generate(inputs=prompts_ids, **gen_kwargs)
        # sequences_ids = torch.cat([sequences_ids, torch.tensor([[self.tokenizer.eos_token_id]]).to(self.config.device_ppo)], dim=-1)
        return sequences_ids[:, prompt_length :]
    
    def generate_ref(self, prompts_ids,temperature):
        # prompts_ids = self.tokenizer.encode(inputs, return_tensors="pt")
        self.ref_model.eval()
        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": self.config.max_response_length,
            # "min_new_tokens": self.config.min_response_length, 
            "_from_model_config": False,
            "logits_processor":[self.logits_processor],
            "temperature":temperature, 
            "do_sample":True
        }
        sequences_ids = self.ref_model.generate(inputs=prompts_ids, **gen_kwargs)
        return sequences_ids

    def get_action(self, sentence):
        pattern = r'^[^\[\]]*\[\s*'  
        pattern += r'^\[\s*(?:0|[1-9]\d*)(?:\s* \s*(?:0|[1-9]\d*))*' 
        pattern += r'\s*\][^\[\]]*$' 
        if not re.fullmatch(pattern, sentence):
            return False, None
        
        match = re.findall(pattern, sentence)
        elements = list(map(int, match[0][1:-1].split(',')))
        return elements, None

    def compute_log_probs(self, sequences, extra_inputs=None):
        actor_logits = self.actor(**sequences, return_dict=True).logits
        ref_logits = self.ref_model(**sequences, return_dict=True).logits
        critic_values = self.critic(**sequences)[-1]
        if extra_inputs is not None:
            extra_loss = self.actor(**extra_inputs, return_dict=True).loss
        else:
            extra_loss = 0.0  
        return actor_logits, critic_values, extra_loss

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    # @staticmethod
    # def load(path, config):
    #     """加载模型"""
    #     tokenizer = AutoTokenizer.from_pretrained(path)
    #     model = AutoModelForCausalLM.from_pretrained(path)
    #     return LLaMAModel(config)