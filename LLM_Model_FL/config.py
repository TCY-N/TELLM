# config.py
import argparse

def get_args():
    # fed参数
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
    parser.add_argument('-episo', '--episode', type=int, default=1000, help='numer of the episode')

    parser.add_argument('-g', '--gpu', type=str, default='2', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-op', '--optimizer', type=str, default='SGD', help='SGD or Adam')
    parser.add_argument('-sc', '--sele_clients', type=int, default=4, help='the number of selected clients')
    parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.05, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=5, help='global model save frequency(of communication)')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')
    parser.add_argument('-time', '--time_lim', type=int, default=15.00, help='required time qos')
    parser.add_argument('-energy', '--energy_lim', type=int, default=2, help='required time qos')
    parser.add_argument('-Band', '--B', type=int, default=2e7, help='required time qos')

    parser.add_argument('-sel', '--select', type=str, default='llama', help='selection method "random","oort", "probing_loss", "llama"')

    parser.add_argument('-Ti', '--Time', type=float, default=15.00, help='except time for Oort')
    parser.add_argument('-En', '--Energy', type=float, default=1.5, help='except time for Oort')
    parser.add_argument('-alpha', '--alpha', type=float, default=10, help='straggler penalty alpha')
    parser.add_argument('-beta', '--beta', type=float, default=2, help='straggler penalty beta')
    parser.add_argument('-epsilon', '--epsilon', type=float, default=0.9, help='epsilon for greedy')
    parser.add_argument('-uni', '--uniform', type=str, default='Di', help='Whether the data set is uniformly distributed and Dirichlet ')
    parser.add_argument('-Di', '--DiSigma', type=float, default=0.15, help='Sigma of Dirichlet distribution')
    parser.add_argument('-ep_delay', '--epsilon_delay', type=float, default=0.8, help='epsilon for greedy')


    # Oort 参数
    # o_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Oort")
    # o_parser.add_argument('-explor_f', 'exploration_factor', type=float, default='0.9', help='exploration factor of oort')
    # o_parser.add_argument('-explor_d', 'exploration_decay', type=float, default='0.98', help='exploration decay of oort')
    # o_parser.add_argument('-explor_min', 'exploration_min', type=float, default='0.2', help='minimum exploration of oort')

    # decision transformer 参数
    dtparser = parser.add_argument_group('decision transformer', 'Configuration for model based')

    dtparser.add_argument('--K', type=int, default=101)
    dtparser.add_argument('--batch_size', type=int, default=1)
    dtparser.add_argument('--state_dim', type=int, default=80)
    dtparser.add_argument('--loss_dim', type=int, default=20)
    dtparser.add_argument('--act_dim', type=int, default=20)
    dtparser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    dtparser.add_argument('--embed_dim', type=int, default=256)
    dtparser.add_argument('--n_layer', type=int, default=6)
    dtparser.add_argument('--n_head', type=int, default=8)
    dtparser.add_argument('--activation_function', type=str, default='tanh')
    dtparser.add_argument('--dropout', type=float, default=0.1)
    dtparser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    dtparser.add_argument('--num_eval_episodes', type=int, default=10)
    dtparser.add_argument('--max_iters', type=int, default=100)
    dtparser.add_argument('--num_steps_per_iter', type=int, default=1000)
    dtparser.add_argument('--device', type=str, default='cuda:2')
    dtparser.add_argument('--train_type', type=str, default='teacher')
    dtparser.add_argument('--samp_prob', type=str, default='linear')
    # parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)


    # PPO_LLM 参数
    ppoparser = parser.add_argument_group('ppo_llm', 'Configuration for ppo and llm')

    ppoparser.add_argument('--max_response_length', type=int, default=8)
    ppoparser.add_argument('--min_response_length', type=int, default=8)
    ppoparser.add_argument('--save_steps', type=int, default=50)
    # ppoparser.add_argument('--model_path', type=str, default="../LLM_finetune/llama1B/pretrained_checkpoints/")
    ppoparser.add_argument('--model_path', type=str, default="../LLM-Research/Llama-3___2-1B-Instruct/")

    ppoparser.add_argument('--actor_peft_path', type=str, default=None)
    ppoparser.add_argument('--critic_peft_path', type=str, default=None)
    # ppoparser.add_argument('--actor_peft_path', type=str, default='save_data/actor_model_ppo02/checkpoint-205/')
    # ppoparser.add_argument('--critic_peft_path', type=str, default='./save_data/critic_model_ppo_norm/checkpoint-205')
    ppoparser.add_argument('--device_ppo', type=str, default='cuda:2')
    ppoparser.add_argument('--report_to', type=str, default='tensorboard')
    ppoparser.add_argument('--gradient_accumulation_steps', type=int, default=100)  #累计batch_size大小 2*100 = 200
    ppoparser.add_argument('--per_device_mini_train_batch_size', type=int, default=1) #一单个minibatch大小 2
    ppoparser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    ppoparser.add_argument('--warmup_steps', type=float, default=0.05)
    ppoparser.add_argument('--traj_batch_llm', type=int, default=1)



    ppoparser.add_argument('--mini_data_buffer_nums', type=int, default=100)  #一组数据step数 至少4条轨迹
    

    ppoparser.add_argument('--gamma', type=float, default=0.99)
    ppoparser.add_argument('--lam', type=float, default=0.99)
    ppoparser.add_argument('--use_advantage_norm', type=bool, default=False)
    ppoparser.add_argument('--kl_penalty_method', type=str, default='mse')
    ppoparser.add_argument('--kl_penalty_beta', type=float, default=0.02)
    ppoparser.add_argument('--value_clip', type=float, default=0.2)
    ppoparser.add_argument('--ratio_clip', type=float, default=0.2)
    
    

    ppoparser.add_argument('--max_iteration', type=int, default=400)
    ppoparser.add_argument('--extra_loss_weight', type=float, default=0.2)
    ppoparser.add_argument('--extra_warmup_steps_ratio', type=float, default=0.2)
    ppoparser.add_argument('--actor_loss_weight', type=float, default=1)
    ppoparser.add_argument('--critic_loss_weight', type=float, default=0.2)
    ppoparser.add_argument('--entropy_beta', type=float, default=0.005)
    # ppoparser.add_argument('--entropy_beta', type=float, default=1)
    ppoparser.add_argument('--max_grad_norm', type=float, default=0.5)
    # ppoparser.add_argument('--max_grad', type=float, default=0.5)
    ppoparser.add_argument('--max_training_timesteps', type=int, default=200000)  # 总收集step， 1000条轨迹
    ppoparser.add_argument('--logging_steps', type=int, default=1)
    ppoparser.add_argument('--ppo_epochs', type=int, default=5)
    ppoparser.add_argument('--max_ep_len', type=int, default=100)
    ppoparser.add_argument('--learning_rate_ppo', type=float, default=5e-5)
    ppoparser.add_argument('--fp16', type=bool, default=True)

    ppoparser.add_argument('--output_dir', type=str, default='./save_data/actor_model_ppo006_debug')
    ppoparser.add_argument('--critic_output_dir', type=str, default='./save_data/critic_model_ppo_debug')
    ppoparser.add_argument('--project_name', type=str, default='grpo_model_nokl_debug')
    ppoparser.add_argument('--CUDA', type=str, default='3')
    ppoparser.add_argument('--reward', type=str, default='real', help='rde real')
    ppoparser.add_argument('--sh_script_path', type=str, default='./grpo_run.sh')
    ppoparser.add_argument('--llm_model', type=str, default='llama', help="gpt, llama")

    # 解析参数
    args = parser.parse_args()


    return args