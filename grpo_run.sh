python  LLM_Model_FL/train_grpo_pro.py \
--fp16 True \
--output_dir './save_data/Mnientropy_actor_model_grpo_list' \
--CUDA 0 \
--project_name 'Mnientropy_grpo_model_nokl_list' \
--mini_data_buffer_nums 800 \
--traj_batch_llm 8 \
--gradient_accumulation_steps 50 \
--save_steps 100 \
--ppo_epochs 5 \
--DiSigma 0.2 \
--max_iteration 200 \
--reward 'rde' \
--entropy_beta 5e-6 \
--learning_rate_ppo 5e-5 \
--sh_script_path ./grpo_run.sh 
# --warmup_steps 15000 \