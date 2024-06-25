# Description

This is the official repository for the paper [Proximal Curriculum with Task Correlations for Deep Reinforcement Learning](https://arxiv.org/pdf/2405.02481), which is published at IJCAI 2024.

## Getting Started

The organization of the repository has the following structure:

```
ijcai2024-proximal-curriculum-target-rl
│   README.md
│   main.py    
│   requirements.txt
│
└───abstract_classes
│       AbstractTeacher.py
│       AbstractGymWrapper.py
│   
└───envs 
│   │
│   └───pointmass
│   │    │    PointMassTeacher.py
│   │    │    PointMassWrapper.py   
│   │    └─── pm_sparse
│   │    │        binary_pointmass.py
│   │    └─── task_datasets
│   │             pm_train.csv
│   │             pm_test_heldout.csv
│   │
│   └───sgr
│   │    │    SgrTeacher.py
│   │    │    SgrWrapper.py
│   │    └─── task_datasets
│   │             sgr_train.csv
│   │             sgr_test_heldout.csv
│   │
│   └───MiniGrid
│             MiniGridTeacher.py
│             MiniGridWrapper.py
│             wrappers.py
│             environments.py
│
└───utils
│   │
│   └───currot_utils
│   │         assignment_solver.py
│   │         buffer.py
│   │         util.py
│   │         wasserstein_interpolation.py
│   │
│   └───custom_mlp
│   │         custom_actor_critic.py
│   │
│   └───helpers4gradient
│             bary_utils.py
│             custom_callback.py
│             monitor.py
│             gradient_utils.py
│             w_encode.py
│
└───teachers_tma
        procurl.py
        proxcorl.py            
```

There are two abstract classes, _AbstractTeacher.py_ and _AbstractGymWrapper_ which contain all the abstract methods that need to be implemented in order to do curriculum experiments for any gym environment. Inside the _envs_ folder there are the 3 subfolders which contains the 3 environment used for the experiments. Each subfolder in addition to the original environment, has concrete implementations of the abstract classes and the datasets used. The _main.py_ can be used to run the different curriculums and environments. In the teachers_tma directory we have extended the AbstractTeacher provided by the TeachMyAgent benchmark with two curriculum strategies, i.e., ProxCoRL and ProxCoRL-un. To run the experiments of BipedalWalker, one can directly use these two teacher interfaces to apply both techniques to the benchmark. 

### Dependencies/Installation

We used Python 3.10 to run the experiments. You can install all the necessary libraries and dependencies from the _requirements.txt_. For CURROT curriculum you need additionally to install the following:
- gurobi optimization for python. You can find documentation and request for licence or free trial here: [Gurobi Optimization](https://www.gurobi.com/). 
- cyipopt. You can find documentation and installation instructions here: [cyipopt](https://cyipopt.readthedocs.io/en/stable/install.html). We recommend to use the conda installation.

Moreover, [Weights and Biases](https://wandb.ai/site) platform is being used for live visualization during training and logging of the experiments. In case you don't want to use it, you can set the argument _--wandb_ to False.

We recommend to use a virtual environment to install the dependencies. 

### Executing program

To run the experiments you can use _main.py_ with the following arguments:
```
usage: main.py [-h] [--option [OPTION]] [--wandb [WANDB]]
               [--tensorboard [TENSORBOARD]] [--env_name [ENV_NAME]]
               [--target_type [TARGET_TYPE]
               [--curriculum [CURRICULUM]] [--seed [SEED]]
               [--beta [BETA]] [--noise [NOISE]]
               [--spdl_threshold [SPDL_THRESHOLD]] 
               [--beta_plr [BETA_PLR]] [--rho_plr [RHO_PLR]
               [--currot_perf_lb [CURROT_PERF_LB]] [--currot_metric_eps [CURROT_METRIC_EPS]]
               [--device [DEVICE]]
               [--model_path [MODEL_PATH]]
```

More details about the arguments are provided below:
 - option: can be set to "train", or "test".
 - wandb: boolean, for using wandb integration. In that case you have to specify your API_KEY.
 - tensorboard, boolean, for using tensorboard logging.
 - env_name: you can set any of the 3 environments, "PointMassSparse", "Sgr", "MiniGrid".
 - target_type: you can set the target types. Each environment has each own pool of available target types. "Sgr" has the following types available: "single-plane", "single-task", "single-model-gaussian", "double-mode-gaussian". "PointMassSparse" has the following types available: "single-task", "single-model-gaussian", "double-mode-gaussian". MiniGrid has the following types available: "single-target". If the target type does not much the available types a value error will be raised.
 - curriculum: any of the curriculum strategies, i.e., "proxcorl", "procurl-val", "iid", "spdl", "plr", "currot", "target".
 - seed: you can specify a random seed to set for the experiments.
 - beta: parameter used in the curriculum strategy.
 - noise: level of noise added to critic values. Used for robustness study.
 - spdl_threshold: performance threshold parameter used in SPDL strategy.
 - beta_plr: temperature for score used in the curriculum strategy for PLR.
 - rho_plr: staleness parameter used in the curriculum strategy for PLR.
 - currot_perf_lb: performance threshold parameter used in the curriculum strategy for CURROT.
 - currot_metric_eps: epsilon parameter used in the curriculum strategy for CURROT.
 - device: you can specify the device to use for training.
 - model_path: if you use test option you can specify the path/paths of the trained model/models.

We provide an example command for starting training with "proxcorl" curriculum on Sgr environment with wandb and tensorboard logging:

```
python main.py --option train --wandb true --tensorboard true --env_name Sgr --target_type single-plane --curriculum proxcorl
```

Tensorboard is already integrated in the experiments to visualize training and validation progress. By running _main.py_ a log folder _run_ with the specific run is created.
If you want to deactivate this option set _--tensorboard_ argument to False. Moreover, the best model is being saved in the _models_ directory that is created.

### Citation

```
@article{tzannetos2024proximal-curriculum-target,
  author  = {Georgios Tzannetos and Parameswaran Kamalaruban and Adish Singla},
  title   = {{P}roximal {C}urriculum with {T}ask {C}orrelation for {D}eep {R}einforcement {L}earning},
  journal = {International Joint Conference of Artificial Intelligence (IJCAI)},
  year    = {2024},
}
```

