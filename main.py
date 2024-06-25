from envs.sgr.SgrWrapper import SgrWrapper, SgrWrapperEval
from envs.sgr.SgrTeacher import SgrTeacher
from envs.pointmass.PointMassWrapper import PointmassWrapper, PointmassWrapperEval
from envs.pointmass.PointMassTeacher import PointMassTeacher
from envs.minigrid.MinigridWrapper import MinigridWrapper, MinigridWrapperEval
from envs.minigrid.MinigridTeacher import MinigridTeacher
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import torch
import argparse
from utils.custom_mlp.custom_actor_critic import CustomActorCriticPolicy

API_KEY = ""  # Specify your API_KEY for using wandb

parser = argparse.ArgumentParser()
parser.add_argument('--option', nargs='?', const=1, type=str, default="train")
parser.add_argument('--wandb', nargs='?', const=1, type=bool, default=False)
parser.add_argument('--tensorboard', nargs='?', const=1, type=bool, default=False)
parser.add_argument('--env_name', nargs='?', const=1, type=str, default="Sgr")
parser.add_argument('--target_type', nargs='?', const=1, type=str, default="single-plane")
parser.add_argument('--curriculum', nargs='?', const=1, type=str, default="proxcorl")
parser.add_argument('--seed', nargs='?', const=1, type=int, default=420)
parser.add_argument('--beta', nargs='?', const=1, type=int, default=110)
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.0)
parser.add_argument('--spdl_threshold', nargs='?', const=1, type=float, default=0.5)
parser.add_argument('--beta_plr', nargs='?', const=1, type=float, default=0.1)  # temperature for score
parser.add_argument('--rho_plr', nargs='?', const=1, type=float, default=0.5)  # staleness parameter
parser.add_argument('--currot_perf_lb', nargs='?', const=1, type=float, default=0.4)
parser.add_argument('--currot_metric_eps', nargs='?', const=1, type=float, default=1.5)
parser.add_argument('--device', nargs='?', const=1, type=str, default="cpu")
parser.add_argument('--model_path', '--list', help="Path of models. Used for testing",
                    nargs='+', type=str, default="models")
args = parser.parse_args()


def main():

    if args.option == "train":
        env_train, env_eval, run, n_steps, \
            gamma, teacher_callback, eval_det, timesteps, \
            eval_freq, env_type = None, None, None, None, None, None, None, None, None, None

        if args.wandb:
            wandb.login(key=API_KEY)

        config = {
            "policy_type": "MlpPolicy",
            "env_name": args.env_name,
            "cur": args.curriculum,
            "beta": args.beta,
            "noise": args.noise,
            "spdl_threshold": args.spdl_threshold,
            "beta_plr": args.beta_plr,
            "rho_plr": args.rho_plr,
            "currot_perf_lb": args.currot_perf_lb,
            "currot_metric_eps": args.currot_metric_eps,
            "target_type": args.target_type
        }

        # Default PPO hyperparams
        learning_rate = 0.0003
        ent_coef = 0
        clip_range = 0.2
        gae_lambda = 0.95
        max_grad_norm = 0.5
        vf_coef = 0.5

        # Training interface for the different environments
        if args.env_name == "PointMassSparse":
            env_type = "binary"
            env_train = PointmassWrapper(cur=config["cur"], env_type=env_type, beta=args.beta,
                                         target_type=args.target_type, metrics=args.wandb,
                                         beta_plr=args.beta_plr, rho_plr=args.rho_plr)
            env_eval = PointmassWrapperEval(env_type=env_type, target_type=args.target_type)
            n_steps = 5120
            gamma = 0.99
            batch_size = 128
            eval_det = False
            eval_freq = 25000
            timesteps = 2000000
            learning_rate = 0.0003
            ent_coef = 0.01
            config["policy_type"] = CustomActorCriticPolicy
            policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, dict(pi=[64], vf=[64])])

        elif args.env_name == "Sgr":
            env_type = "binary"
            env_train = SgrWrapper(cur=config["cur"], beta=args.beta,
                                   target_type=args.target_type, metrics=args.wandb,
                                   beta_plr=args.beta_plr, rho_plr=args.rho_plr)
            env_eval = SgrWrapperEval(target_type=args.target_type)
            n_steps = 5120
            batch_size = 256
            gamma = 0.99
            timesteps = 1000000
            eval_freq = 25000
            eval_det = False
            config["policy_type"] = CustomActorCriticPolicy
            policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[64, dict(pi=[32], vf=[32])])

        elif args.env_name == "MiniGrid":

            policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 128, 64, 32])
            n_steps = 25600
            timesteps = 4_000_000
            eval_freq = 100_000
            gamma = 0.99
            batch_size = 64
            ent_coef = 0.01
            eval_det = False
            env_train = MinigridWrapper(cur=config["cur"], beta=args.beta,
                                        target_type=args.target_type, metrics=args.wandb,
                                        beta_plr=args.beta_plr, rho_plr=args.rho_plr)

            env_eval = MinigridWrapperEval(target_type=args.target_type)
            config["policy_type"] = "MlpPolicy"

        else:
            raise NotImplementedError(f"Environment {args.env_name} not implemented")

        config["n_steps"] = n_steps
        config["gamma"] = gamma
        config["total_timesteps"] = timesteps

        if args.wandb:
            run = wandb.init(
                settings=wandb.Settings(start_method='thread'),
                project=f"{config['env_name']}",
                config=config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=False,  # optional
                group=f'{config["cur"]}',
                name=f'{config["cur"]}',
                mode="online",  # disabled, online, offline
            )
            # Metrics are defined
            wandb.define_metric("env_n_calls")
            wandb.define_metric("global_env_steps")
            wandb.define_metric("global_step")
            wandb.define_metric("context1", step_metric="global_step")

        if args.target_type == "single-task":
            num_evals = 100
        else:
            num_evals = 10
        # Use deterministic actions for evaluation
        if args.wandb:
            eval_callback = EvalCallback(env_eval, eval_freq=eval_freq, deterministic=eval_det,
                                         best_model_save_path=f"models/{run.id}",
                                         n_eval_episodes=num_evals*env_eval.num_of_tasks,
                                         render=False)
        else:
            eval_callback = EvalCallback(env_eval, eval_freq=eval_freq, deterministic=eval_det,
                                         best_model_save_path=f"models/example",
                                         n_eval_episodes=num_evals*env_eval.num_of_tasks,
                                         render=False)

        # Interface for Teacher CallBack
        if args.env_name == "PointMassSparse":
            teacher_callback = PointMassTeacher(env=env_train, type_env=env_type, target_type=args.target_type, cur=config["cur"], eps=args.noise,
                                                n_steps=config["n_steps"],
                                                spdl_pthresh=args.spdl_threshold, Na=200000, alpha=0.01, epsilon=0.5,
                                                currot_perf_lb=args.currot_perf_lb, currot_metric_eps=args.currot_metric_eps,
                                                metrics=args.wandb, verbose=1)

        elif args.env_name == "Sgr":
            teacher_callback = SgrTeacher(env=env_train, type_env=env_type, target_type=args.target_type, cur=config["cur"], eps=args.noise,
                                          n_steps=config["n_steps"],
                                          spdl_pthresh=args.spdl_threshold, Na=0, alpha=0.0002, epsilon=0.2,
                                          currot_perf_lb=args.currot_perf_lb, currot_metric_eps=args.currot_metric_eps,
                                          metrics=args.wandb, verbose=1)
        elif args.env_name == "MiniGrid":
            teacher_callback = MinigridTeacher(env=env_train, type_env=env_type, target_type=args.target_type, cur=config["cur"], eps=args.noise,
                                               n_steps=config["n_steps"],
                                               spdl_pthresh=args.spdl_threshold, Na=0, alpha=0.0002, epsilon=0.2,
                                               currot_perf_lb=args.currot_perf_lb, currot_metric_eps=args.currot_metric_eps,
                                               metrics=args.wandb, verbose=1)

        if args.wandb:
            run_path = f"runs/{run.id}"
            if not args.tensorboard:
                run_path = None
            #
            model = PPO(config["policy_type"], env_train, verbose=2, policy_kwargs=policy_kwargs,
                        gamma=config["gamma"], n_steps=config["n_steps"], device=args.device, learning_rate=learning_rate,
                        batch_size=batch_size, seed=args.seed, n_epochs=10, tensorboard_log=run_path, ent_coef=ent_coef,
                        clip_range=clip_range, gae_lambda=gae_lambda, max_grad_norm=max_grad_norm, vf_coef=vf_coef)

            model.learn(total_timesteps=config["total_timesteps"],
                        callback=[eval_callback, teacher_callback, WandbCallback(verbose=2)])

        else:
            run_path = "runs/example"
            if not args.tensorboard:
                run_path = None
            model = PPO(config["policy_type"], env_train, verbose=2, policy_kwargs=policy_kwargs,
                        gamma=config["gamma"], n_steps=config["n_steps"], device=args.device,
                        tensorboard_log=run_path, seed=args.seed, n_epochs=10)

            model.learn(total_timesteps=config["total_timesteps"],
                        callback=[eval_callback, teacher_callback])

        if args.wandb:
            run.finish()

    elif args.option == "test":
        env_eval = None
        # Also specify some learning parameters that might be different for each env
        if args.env_name == "PointMassSparse":
            env_type = "binary"
            env_eval = PointmassWrapperEval(env_type=env_type, target_type=args.target_type)
        elif args.env_name == "Sgr":
            env_eval = SgrWrapperEval(target_type=args.target_type)

        num_evals = 10
        # Load model
        path = args.model_path  # Should be a valid path/list of paths that contains a model.zip
        for mdl_path in path:
            trained_model = PPO.load(f"models/{mdl_path}/best_model.zip", env_eval, device=args.device)
            mean_reward, std_reward = evaluate_policy(trained_model, env_eval, n_eval_episodes=num_evals * env_eval.num_of_tasks)
            print(f"Mean reward for model {mdl_path} is {mean_reward} with standard deviation {std_reward}")


if __name__ == "__main__":
    main()
