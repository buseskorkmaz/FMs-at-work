# Foundation Models at Work: Fine-Tuning for Fairness Beyond Human Feedback
 
Official code from the paper "Foundation Models at Work: Fine-Tuning for Fairness Beyond Human Feedback", based on the implementation of ILQL [arxiv] (https://arxiv.org/abs/2206.11871) .
 
# Setup
 
### **Preprocessed Data**

Copy the data and indexes to under `/data/task_path` in your environment. For workable, first move the preprocessed data files into your `/data` directory: 

```shell
cp -r /dccstor/autofair/workable_processed/candidates_w_main_location path_to_env/data/workable_rl_dataset
cp -r /dccstor/autofair/workable_processed/job_descriptions_w_q_prompt_eng path_to_env/data/workable_rl_dataset
```

Then copy the indexes:

```shell
cp /dccstor/autofair/workable_processed/train_idxs.json path_to_env/data/workable_rl_dataset
cp /dccstor/autofair/workable_processed/test_idxs.json path_to_env/data/workable_rl_dataset
cp /dccstor/autofair/workable_processed/eval_idxs.json path_to_env/data/workable_rl_dataset
```

### **Dependencies and PYTHONPATH**
 
This repo was designed for python 3.9.7
 
```shell
pip install -r requirements.txt
export PYTHONPATH="$PWD/src/"
```

# Inference

Copy the checkpoint to under `/outputs/task_name`. For workable:

```shell
cp -r /dccstor/autofair/workable_processed/checkpoints/conditional_workable_official_iql_bc_test1_16383_eng/ path_to_env/outputs/workable/
```

Edit the config file (`config/workable/eval_policy.yaml`) with input/output paths and hyperparameter values. Then, evaluate policy:

```shell
cd scripts/eval/workable
jbsub -queue x86_24h -mem 32g -cores 4+1 python eval_policy.py 
```

Distill policy to see original and rewrite diversity scores:

```shell
cd scripts/eval/workable
python distill_policy_eval.py --eval_file ../../../outputs/workable/your_output_path/eval_logs.pkl 
```

Impact ratio and diversity score calculations for both original and rewritten job descriptions (in CCC):

```shell
cd scripts/eval/workable
jbsub -queue x86_24h -mem 32g -cores 4+1 python impact_ratio_calc.py --eval_file ../../../outputs/workable/your_output_path/eval_logs.pkl --save_path ../../../outputs/
```

# Running Experiments
 
`scripts/` contains all experiment scripts. To run any script in `scripts/`:
1. Navigate to the script's directory.
2. `python script_name.py`
 
Optional:
* Edit the config file corresponding to the script as you desire.
* Provide commandline args [hydra](https://hydra.cc/docs/intro/) style like: `python script_name.py eval.bsize=5 train.lr=1e-6 wandb.use_wandb=false`
* Run data parallel training or evaluation on multiple GPUs like: `python -m torch.distributed.launch --nproc_per_node [N_GPUs] --use_env script_name.py arg1=a arg2=b`
 
By default all training scripts log to wandb. To turn this off, set `wandb.use_wandb=false` in the training config.
 
### **Recommended Experiment Workflow:**
 
Here I outline a recommended workflow for training offline RL agents. Suppose that I want to train a bunch of different offline RL agents to generate job descriptions with the diversity reward.
 
I would first train a BC model on the data:
 
``` shell
cd scripts/train/hackernews/
python train_bc.py
```
 
Then convert this BC checkpoint into one compatible with the offline RL models:
 
``` shell
cd ../data/
python convert_bc.py --load ../../outputs/hackernews/conditional_hackernews_official_bc_test1/model.pkl --save ../../outputs/hackernews/conditional_hackernews_official_bc_test1/model_converted.pkl
```
 
Then edit the checkpoint that offline RL is configured to train with:
 
``` shell
cd ../train/
python train_iql.py model.load.checkpoint_path=outputs/hackernews/conditional_hackernews_official_bc_test1/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.0
```
 
This is just one workflow though, you can also train the BC model at the same time as the offline RL agent by setting `train.loss.awac_weight=1.0` in the training config.
 
# Repo Overview
 
* All data is provided pre-processed in the `data/` folder.
* `scripts/` contains all scripts for running training, evaluation, and data pre-processing steps in the paper. Scripts are organized into subfolders corresponding to the dataset used.
* `config/` contains .yaml configs for each script. This repo uses [hydra](https://hydra.cc/docs/intro/) to manage configs. Configs are organized into subfolders corresponding to the dataset used. Most config files are named the same as their corresponding script, but if you are unsure which config corresponds to a script, check the line `@hydra.main(config_path="some_path", config_name="some_name")` to see which config file the script corresponds to.
* `src/` contains all the core implementations. See `src/models/` for all model implementations. See `src/data/` for all base data processing and MDP abstraction code. See `src/utils/` for various utility functions. See `src/hackernews/` for hackernews hiring dataset specific code respectively.
* `ILQL` is referred to as `iql` throughout the repo.
 
## Config Framework Overview
 
Each script is associated with a config file. The config file specifies which models, dataset, and evaluators are to be loaded by the script and their corresponding hyperparameters. See `configs/hackernews/train_iql.yaml` for an example.
 
Each possible model, dataset, or evaluator object is given its own config file, which specifies default values for that object and a special `name` attribute, which tells the config manager what class to load. See `configs/hackernews/model/per_token_iql.yaml` for an example.
 
The files `src/load_objects.py` and `src/hackernews/load_objects.py` define how each object is loaded from its corresponding config. The `@register('name')` tag above each load object function links to the `name` attribute in the config.
 
You may notice a special `cache_id` attribute associated with some objects in a config. For an example, see `train_dataset` in `configs/hackernews/train_iql.yaml`. This attribute tells the config manager to cache the first object that it loads that is associated with this id, and then to return this cached object for subsequent object configs with this `cache_id`.
 
For all configs, use paths relative to the repo root.
 
## A Few Abstractions to be Aware of
 
Hackernews and Workable implement a few base classes. Once implemented, all the offline RL algorithms can be applied to the task in a plug-and-play manner. See the "Creating Your Own Tasks" section for an overview of what should be implemented in order to create your own tasks. Below, we outline the key abstractions that make this possible.
 
* `data.language_environment.Language_Environment` – represents a task POMDP environment, which a policy can interact with. It has a gym-like interface.
* `data.language_environment.Policy` – represents a policy which can interact with an environment. Each of the offline RL algorithms in `src/models/` has a corresponding policy.
* `data.language_environment.Language_Observation` – represents a text observation that is returned by the environment and given as input to a policy.
* `data.language_environment.interact_environment` – a function which takes in an environment, a policy, and optionally the current observation and runs an environment interaction loop. If the current observation is not provided, it automatically fetches an initial state by resetting the environment.
* `data.rl_data.DataPoint` – defines a standardized data format that is fed as input to all offline RL agents on all tasks. These data structures are created automatically from a given `Language_Observation`.
* `data.rl_data.TokenReward` – defines a reward function given at every single token, which can be used for learning more fine grained control. This is provided on top of the environment's reward, which comes not at every token but instead after each turn of interaction. In all our experiments we set this reward to a constant 0, such that it has no effect.
* `data.tokenizer.Tokenizer` – specifies how to convert strings to and from sequences of tokens which can then be fed as input to language models.
* `data.rl_data.RL_Dataset` – defines a dataset object which returns `DataPoint` objects and is used for training offline RL agents. There are two versions of `RL_Dataset`:
   1. `List_RL_Dataset`
   2. `Iterable_RL_Dataset`

 
# Creating Your Own Tasks
 
Hacker News hiring task has a corresponding environment and dataset implemented in the codebase, as described above. 

You can similarly define your own tasks that can easily be run on all these offline RL algorithms. This codebase implements a simple set of RL environment abstractions that make it possible to define your own environments and datasets that can plug-and-play with any of the offline RL algorithms.

All of the core abstractions are defined in `src/data/`. Here we outline what needs to be implemented in order to create your own tasks. For examples, see the implementations in `src/hackernews/`.
 
## 1. Create an environment and define observations:
 
All tasks must implement subclasses of: `Language_Observation` and `Language_Environment`, which are in `src/data/language_environment.py`.
 
### **`Language_Observation`:**
This class represents the observations from the environment that will be input to your language model.
 
A `Language_Observation` must define the following two functions.
 
---
 
#### **`to_sequence`**
 
``` python
def to_sequence(self) -> Tuple[List[str, Optional[float]], bool]:
```
 
**Description:**
 
A function which converts the observation object into a standard format that can be input to the language model and used for training.
 
**Returns:**
1. a list of (utterance, reward) tuples. The tuples are meant to represent alternating environment interactions: your agent's utterance and the environment's response. Utterances corresponding to the environment response should have reward=None, and those corresponding to the agent's utterances should have reward=some_float.
2. a boolean indicating whether this observation is the last one in the interaction.
 
#
 
#### **`__str__`**
 
``` python
def __str__(self) -> str:
```
 
**Description:**
 
This is only used to print the observation to the terminal. It should convert the observation into some kind of string that is interpretable by a user.
 
**Returns:** a string.
 
---
 
### **`Language_Environment`:**
This class represents a gym-style environment for online interaction, which is only used for evaluation.
 
A Language_Environment must define the following three functions.
 
---
 
#### **`step`**
 
``` python
def step(self, action: str) -> Tuple[Language_Observation, float, bool]:
```
 
**Description:**
 
Just like a standard gym environment, given an action in the form of a string, step the environment forward.
 
**Returns:** a tuple of (Language_Observation, reward, terminal).
 
#
 
#### **`reset`**
 
``` python
def reset(self) -> Language_Observation:
```
 
**Description:**
 
This resets the environment to an initial state.
 
**Returns:** the corresponding initial `Language_Observation`
 
#
 
#### **``is_terminal``**
 
``` python
def is_terminal(self) -> bool:
```
 
**Description:**
 
Outputs whether the environment has reached a terminal state.
 
**Returns:** a boolean indicating if the environment has reached a terminal state.
 
---
 
### **2. Create a Dataset:**
 
All tasks must implement subclasses of either `List_RL_Dataset` or `Iterable_RL_Dataset` or both, which are defined in `src/data/rl_data.py`.
 
### **`List_RL_Dataset`:**
This class represents a list dataset (or an indexable dataset of finite length) that can be used to train offline RL agents.
 
A `List_RL_Dataset` must define the following two functions.
 
---
 
#### **`get_item`**
 
``` python
def get_item(self, idx: int) -> DataPoint
```
 
**Description:**
 
This gets an item from the dataset at a given index.
 
**Returns:** a `DataPoint` object from the dataset.
 
#
 
#### **``size``**
 
``` python
def size(self) -> int
```
 
**Description:**
 
Returns the size of the dataset.
 
**Returns:** the dataset's size.
 
---
 
### **`Iterable_RL_Dataset`:**
This class represents an iterable dataset (or a non-indexable dataset that stochastically samples datapoints i.i.d.) that can be used to train offline RL agents.
 
A `Iterable_RL_Dataset` must define the following function.
 
---
 
#### **``sample_item``**
 
``` python
def sample_item(self) -> DataPoint
```
 
**Description:**
 
Samples a datapoint from the dataset.
 
**Returns:** a `DataPoint` object from the dataset.
 
---
 

