from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from peft import PeftModel, PeftConfig
import re
import copy
from transformers import LogitsProcessor
from datasets import load_dataset

class LlamaModel():
    def __init__(self, model_file, lora_model, dev, args):
        self.args = args
        self.dev = dev
        self.model = self._getmodel(model_file, lora_model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_file)
        self._gettoken(model_file)
        self.allowed_token_ids = self._limit_token()
        

    def _limit_token(self):
        allowed_tokens = [str(i) for i in range(self.args['num_of_clients'])]  # 允许的token（需转换为token ID）
        # allowed_tokens.remove("4")
        # allowed_tokens.extend(['Ġ','[',']'])
        allowed_token_ids = self.tokenizer.convert_tokens_to_ids(allowed_tokens)
        return allowed_token_ids 
    
    def logits_processor(self, input_ids, scores):
        # 只能能生成指定token (0~19 ' ', [])
        mask = torch.ones_like(scores) * -float("inf")
        mask[:, self.allowed_token_ids] = 0  # 仅允许的token的logits保留
        return scores + mask
        
    def _getmodel(self,model_file,lora_model):
        model = AutoModelForCausalLM.from_pretrained(model_file)
        model = PeftModel.from_pretrained(model, lora_model)
        print(f'load successfully from {lora_model}')
        model.to(self.dev)
        return model

    def _gettoken(self, model_file):
        # tokenizer = AutoTokenizer.from_pretrained(model_file) # 加载tokenizer
        if self.args['llm_model'] == 'llama':
            self.tokenizer.pad_token = "<|reserved_special_token_9|>"
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def generate_limit(self, input_text):     
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.dev)
        prompt_length = inputs.shape[1]
        processor = ConstrainedLogitsProcessor(self.allowed_token_ids, prompt_length) 
        self.model.eval()
        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": 10,
            "min_new_tokens": 10, 
            "_from_model_config": False,
            "logits_processor":[self.logits_processor],
            "temperature":5, 
            "do_sample":True
        }
    
    def generate(self, input_text,temperature):     
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.dev)
        prompt_length = inputs.shape[1]
        maxlenght = self.args['sele_clients'] * 2
        processor = ConstrainedLogitsProcessor(self.allowed_token_ids, prompt_length, 220, maxlenght) 
        self.model.eval()
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
        # sequences_ids = self.actor.generate(inputs=prompts_ids, **gen_kwargs, temperature=temperature, do_sample=True)
        outputs = self.model.generate(inputs,  **gen_kwargs)

        # generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        generated_text = outputs[:, prompt_length :]
        # generated_text = generated_text.split('=>')[1]
        return generated_text
    
    def generate_multi(self, input_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        inputs = inputs.to(self.dev)
        outputs = self.model.generate(inputs, max_length=3072, 
                                        top_k=10,temperature=0.8,
                                        do_sample=True,
                                        num_return_sequences=15)
        generated_text = []
        for i, seq in enumerate(outputs):
            txt = self.tokenizer.decode(seq, skip_special_tokens=True)
            # txt = f"Result {i+1}: {txt.split('=>')[1]}"
            txt = txt.split('=>')[1]
            generated_text.append(txt)
        return generated_text
    
    
    def check2action(self, sentence,  client_num, select_num):  
        pattern = r'^[^\[\]]*\[\s*'  
        pattern += r'(?:0|[1-9]\d*)(?:\s* \s*(?:0|[1-9]\d*))*'  
        pattern += r'\s*\][^\[\]]*$'  
        if not re.fullmatch(pattern, sentence,re.DOTALL):
            return False, None

        list_parttern = r'\[\s*' + r'(?:0|[1-9]\d*)(?:\s* \s*(?:0|[1-9]\d*))*' + r'\s*\]'
        match = re.findall(list_parttern, sentence)
        if not match:
            return False, None
        try:
            elements = list(map(int, match[0][1:-1].split()))
        except ValueError:
            return False, None

        if len(elements) != len(set(elements)):
            return False, None

        if len(elements) != select_num:
            return False, None

        if any(not (0 <= x < client_num) for x in elements):
            return False, None
        
        return True, elements


    def txt2data_cot(self, text):
        right_ans = []
        right_return = []
        for i, val in enumerate(text):
            result, returns =  self.check_text_format(val)
            if result != None:
                right_ans.append(copy.deepcopy(result))
                right_return.append(copy.deepcopy(returns))
        if len(right_ans)<2:
            return None,None
        max_acc_index = max(enumerate(right_return), key=lambda item: item[1]['reward'])[0]
        return right_ans[max_acc_index], right_return[max_acc_index]
    
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

    def process2txt(self, last_inf, round, state, client_num, comm_num, qos,  ref, llm_model, band=2e7):
        input = "As a federated training agent, you are responsible for selecting the most suitable clients from the device pool to optimize performance in the current round.\n"
        input += f"Select {comm_num} indexes from given {client_num} clients to participate in federated learning training based on the return information in last round and the clients\' states.\n"
        round_inf = f"The total bandwidth of the system is {band:.2e}. The reference QoS time is {qos:.2f}s. The current communication is round {round}.\n"
        input += last_inf + round_inf
        input += 'The states information of each client is shown in the following table\n'

        # round_inf = f"The current communication is round {round}.\n"
        # input += "For last round, " + last_inf+round_inf

        state_inf = ''
        if llm_model == 'llama':
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
        else:
            input_head = "| client index | max power | max frequency | channel gain | number of compute cycles | data size | local loss | local loss after trained | inner-product betweent local model and global | percentage of same sign betweent local model and global model | last selected round | selected times |\n| --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            for key, val in enumerate(state):
                state_inf += f"| {chr(key+65)} |"
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
                state_inf += f" {self.val_form(val['loss_trained'])} |"
                state_inf += f" {self.val_form(val['inner_product'])} |"
                state_inf += f" {self.val_form(val['sign'])} |"
                state_inf += f" {self.val_form_int(val['round'])} |"
                state_inf += f" {self.val_form_int(val['times'])} |\n"
                # state_inf += f"the selected times is {val['times']}, "
                # state_inf += f"and the disent to the global model is {val['disent']}.\n"
            state_inf += '\n'
    # history_action = ''
        # if actions is not None:
        #     history_action += 'The historical actions for selection of each round are as follows\n'
        #     for i, action in enumerate(actions):
        #         history_action += f'round{i}:{action}\n'

        notice = f"Please select {comm_num} indexes from given {client_num}  client indexes as a result.\n"
        format_ref = f'Give one result directly. Do not ouput anything else. Your response consists only of indexes.'# Ensure that your response is concise and brief and only contains one indexes list.\nFormat of respond: {ref}.\n'
        input += input_head + state_inf + notice + format_ref 

        return input
    
    def create_return(self, round, data):
        if round == 1:
            return None
        returns = {}
        for key, val in data.items():
            if key != 'state':
                returns[key] = val
        return returns
    
    def update_last(self, data):
        # r = f"the return of round {round} is {data['reward']:.4f}, "
        t = f"the time consumpsion of round {round} is {data['time']:.2f}, "
        e = f"the energy consumpsion of round {round} is {data['energy']:.4f}, "
        acc = f"the accuracy of round {round} is {data['acc']:.4f}.\n"
        last_inf = t+e+acc
        return last_inf
    
    def preprocess(self, last_inf, rounds, state, client_num, comm_num, qos, ref, llm_model):
        # returns = self.create_return(round, data)
        inputs = self.process2txt(last_inf, rounds, state, client_num, comm_num, qos, ref, llm_model)
        return inputs
    

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
        

if __name__=="__main__":

    MODEL_PATH = "./LLM-Research/Llama-3___2-1B-Instruct"
    # MODEL_PATH = "../llama1B/pretrained_checkpoints"
    # llama = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    lora_path = '../llama1B/llama_FL/llama_FL300000'  
    llama = LlamaModel(MODEL_PATH, lora_path,'cuda:0')

    while True:
        with open('input.txt', 'r') as file:
            input_text = file.read()
            # input_text = input_text.replace('\\n', '\n')
        # index = int(input())
        # ca = books['train'][index]

        # 测试生成
        # input_text =ca['input'] + '=>'

        print(f"标签内容是：{input_text}")

        allowed_tokens = [str(i) for i in range(20)] 
        # allowed_tokens.extend(['Ġ','[',']'])
        allowed_token_ids = llama.tokenizer.convert_tokens_to_ids(allowed_tokens)

        def logits_processor(inputs_ids, scores):
            mask = torch.ones_like(scores) * -float("inf")
            mask[:, allowed_token_ids] = 0  
            return scores + mask
        
        inputs_ids = llama.tokenizer.encode(input_text, return_tensors="pt").to(llama.dev)
        prompt_length = inputs_ids.shape[1]
        processor = ConstrainedLogitsProcessor(allowed_token_ids, prompt_length, 220, ) 

        gen_kwargs = {
            "pad_token_id": llama.tokenizer.pad_token_id,
            "eos_token_id": llama.tokenizer.eos_token_id,
            "bos_token_id": llama.tokenizer.bos_token_id,
            "max_new_tokens": 8,
            "min_new_tokens": 8, 
            "_from_model_config": False,
        }

        output = llama.model.generate(
            inputs_ids,
            **gen_kwargs,
            # logits_processor=[logits_processor],
            # logits_processor=[processor],
            temperature=1, do_sample=True
        )
        generated_text = llama.tokenizer.decode(output[0] , skip_special_tokens=True)
        generation = output[0, prompt_length:]
        tokens = llama.tokenizer.convert_ids_to_tokens(generation)
        # generated_text = llama.generate(input_text)

        with open('ouput_test.txt', 'a+') as file:
            file.write(''.join(generated_text))
            file.write("\n====================\n")
        # res = llama.txt2data(generated_text)

        # action, returns = llama.txt2data_cot(generated_text)

        print(generated_text)
        # print(res)