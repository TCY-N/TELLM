# TELLM


This repository, corresponding to the paper [Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning](https://ieeexplore.ieee.org/document/11303187). T-ELLM leverages natural language-based scenario prompts to enhance generalization across varying network conditions. The framework decouples the joint optimization problem mathematically, enabling tractable learning of device selection policies while delegating resource allocation to convex optimization tools. To facilitate the evolutionary process, T-ELLM interacts with a sample-efficient, model-based virtual learning environment that captures the relationship between device selection and learning performance. This developed virtual environment reduces reliance on real-world interactions, thus minimizing communication overhead while refining the LLM-based decision-making policy through group relative policy optimization.  Experimental results demonstrate that T-ELLM outperforms benchmark methods in energy efficiency and exhibits robust adaptability to environmental changes.

Note that this is a research project and by definition is unstable. Please write to us if you find something not correct or strange. We are sharing the codes under the condition that reproducing full or part of codes must cite the paper.

## Installation

```
conda env create -f environment.yml
conda activate TELLM
```

## Dataset Preparation

The complete T-ELLM requires training a **model-based virtual learning environment** first.  **offline dataset** can be collected from the general FL process. The dataset should include the following NumPy files:

- `accs.npy` – accruacy sequences
- `states.npy` – state trajectories
- `loss.npy` – FL loss sequences
- `actions.npy` – action sequences

All files must be placed inside the `DT_ModelBased/training_data/` folder at the project root. The folder structure should look like:

```
.
├── data
│   └── MNIST
├── DT_ModelBased      # vitural enviroment, model training
│   ├── decision_transformer
│   ├── experiment_state.py
│   └── training_data
│       ├── accs.npy
│       ├── actions.npy
│       ├── loss.npy
│       └── states.npy
├── FL_LLM_run.sh 
├── grpo_run.sh
├── LLM_Model_FL  # LLM training and testing
├── LLM-Research   # LLM Base Model
├── model_check   # Lora\ model checkpoint
│   └── checkpoint-1000g20model
└── original_data_1000_20.mat  # weirless environment config data
```

Make sure the `.npy` files are consistent in episode length and dimension across all modalities before running training scripts.

After training is completed, the checkpoint of virtual model is saved in `/DT_ModelBased/decision_transformer/major_save_data` and read by `LLM_Model_FL/env_reward.py`. 

Make sure the **virtual model** and **base model** can be correctly read before training the LLM.

## Training

To start training model-based virtual learning environment ：

``` 
python ./DT_ModelBased/experiment_state.py
```

To start training **T-ELLM**, first config the training parammeters  in `./LLM_Model_FL/config.py`.

Some parameter has config in script, then run the following command to begin training, for `GRPO` :

```
sh grpo_run.sh
```

for `PPO`, set the first line

```
python LLM_Model_FL/train_pro.py
```

## Usage

The main file for the test program is `LLM_Model_FL/LLM_reason.py`, and the environment configuration can also be modified in `./LLM_Model_FL/config.py`.

To evaluate the trained model, run the following command:

```
sh FL_LLM_run.sh
```


## Reference

```bibtex
@ARTICLE{11303187,
  author={Tan, Chongyang and Wen, Ruoqi and Li, Rongpeng and Zhao, Zhifeng and Hossain, Ekram and Zhang, Honggang},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning}, 
  year={2026},
  volume={44},
  number={},
  pages={2904-2921},
  keywords={Wireless communication;Training;Resource management;Optimization;Virtual environments;Performance evaluation;Decision making;Costs;Wireless sensor networks;Data models;Large language model;generative policy;wireless federated learning;resource management;convex optimization;reinforcement learning},
  doi={10.1109/JSAC.2025.3645754}}

```
## Acknowledgements
This implementation is based on code from several repositories.
[decision-transformer.](https://github.com/kzl/decision-transformer)
[LLM-RLHF-Tuning](https://github.com/Joyce94/LLM-RLHF-Tuning)
