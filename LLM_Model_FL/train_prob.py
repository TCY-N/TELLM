import os
import config
args = config.get_args()
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA  

import config
from llm_model import LLaMAModel
from env_reward import  FL_reward
from PPO_trainer_prob import *
import scipy.io as scio




def main():
    datafile = 'original_data_1000_{}.mat'.format(args.num_of_clients)
    env_config  = scio.loadmat(datafile)


    ppo_model = LLaMAModel(args)
    reward_env = FL_reward(args, env_config)
    ppo_trainer = PPOTrainer(args, ppo_model, reward_env)
    ppo_trainer.train()

if __name__ == "__main__":
    main()