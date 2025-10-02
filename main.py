import argparse
import pickle
from datetime import datetime
import icem_utils as utils
from agent import Agent
from dotmap import DotMap
from fake_env import FakeEnv
from gym_env import GymEnv
from icem import Icem
from mbop import MBOP
import numpy as np
import torch

ALG_ICEM = "alg_icem"
ALG_MBOP = "alg_mbop"
ALG_BC = "alg_bc"


class BC_policy:
    def __init__(self, path, obs_mask):
        self.ensemble = pickle.load(open(path, "rb"))
        self.obs_mask = obs_mask

    def get_action(self, s, a_prior, **kwargs):
        preds = []
        s = (s * self.obs_mask)[np.newaxis, ...]
        a_prior = a_prior[np.newaxis, ...]
        for policy in self.ensemble:
            preds.append(policy.predict(s, a_prior)[0])
        return np.mean(np.array(preds), axis=0), None, None

    def get_action_gpu(self, s, a_prior, use_obs_mask):
        preds = []
        if use_obs_mask:
            s = (s * torch.Tensor(self.obs_mask).to(s.device))[np.newaxis, ...]
        else:
            s = s[np.newaxis, ...]
        a_prior = a_prior[np.newaxis, ...]
        for policy in self.ensemble:
            preds.append(policy.predict(s, a_prior, to_cpu=False)[0])
        return torch.mean(torch.stack(preds), dim=0), None, None


def load_reward_fns(module_name):
    reward_module = utils.load_reward_module(module_name)
    fn_names = ["reward_fn_gt", "reward_fn", "termination_fn", "default_obs_mask"]
    dot_names = ["r_gt", "r", "t", "obs_mask"]
    fns = {}
    for i in range(len(fn_names)):
        if hasattr(reward_module, fn_names[i]):
            fns[dot_names[i]] = getattr(reward_module, fn_names[i])
        elif fn_names[i] != "reward_fn_gt":
            print(f"Warning: module {module_name} does not contain function {fn_names[i]}")
    return DotMap(fns)


# Setup fake environment
def setup_model(config, r_fns, q_fn, bc_policy, threshold, q_k):
    if not config.ground_truth:
        if "dataset" not in config.keys():
            raise Exception("Dataset must be included (--dataset)")
        print("Using learned model of the dynamics")
        models = pickle.load(open(config.model, "rb"))
        if type(models) is tuple and len(models) == 2:
            (models, _) = models
        config.use_mean = True
        fake_env = FakeEnv(models, r_fns.r, r_fns.t, config, q_fn, bc_policy, k=q_k)
        (dataset, metadata) = pickle.load(open(config.dataset, "rb"))
        # Update settings, get elite members and compute threshold
        if threshold is not None:
            fake_env.setup(dataset, config.pessimism_coef, metadata, thresh=threshold)
        else:
            fake_env.setup(dataset, config.pessimism_coef, metadata)
        return fake_env
    return None


def main(args):
    t_start = datetime.now().replace(microsecond=0)
    config = utils.load_config_with_args(args)
    utils.print_config(config)
    utils.set_seeds(config["seed"])
    config.log_path = utils.setup_log(config.log_dir, config.log_name)

    # Load reward functions
    reward_fns = load_reward_fns(config.reward_module)

    # Load Q-function
    q_fn = None
    if args.q_fn is not None:
        print("Loading Q-function from", args.q_fn)
        q_fn = pickle.load(open(args.q_fn, "rb"))

    # Setup environment
    env = GymEnv(config.env, seed=config.seed, reward_fn=reward_fns.r_gt, obs_mask=reward_fns.obs_mask, v_fns=q_fn)
    config.obs_dim = env.get_observation_dimension()
    config.act_dim = env.get_action_dimension()

    # Load BC policy
    bc_policy = None
    if args.bc_policy is not None:
        print("Loading BC policy from", args.bc_policy)
        bc_policy = BC_policy(args.bc_policy, reward_fns.obs_mask)

    config.min_max_action_value = env.get_min_max_action_value()

    # Setup fake environment
    model = setup_model(config, reward_fns, q_fn, bc_policy, args.threshold, args.q_k)

    # Initialize agent and iCEM parameters
    agent = Agent(config)

    if config.alg == ALG_ICEM:
        policy = Icem(config)
    elif config.alg == ALG_MBOP:
        if model.bc_policy is None:
            raise Exception("MBOP algorithm must have a BC policy (--bc_policy)")
        policy = MBOP(config)
    elif config.alg == ALG_BC:
        policy = bc_policy
    else:
        print("Error: unknown algorithm", args.alg)
        exit(1)

    logs = []
    for i in range(config.n_episodes):
        print("\nEpisode:", i)
        log = agent.rollout(env, policy, model, config.seed + i, config.n_episodes > 1)
        logs.append(log)
        if config.n_episodes > 1:
            utils.print_exp_metrics(logs, i)
        print("Saving logs to", config.log_path)
        pickle.dump(logs, open(config.log_path, "wb"))
    t_exp = datetime.now().replace(microsecond=0) - t_start
    print(f"\ntime\t{t_exp}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iCEM with MOReL")
    parser.add_argument("--config", "-c", type=str, required=True, help="path to config file with exp params")
    parser.add_argument("--n_episodes", type=int, help="number of episodes")
    parser.add_argument("--population_size", type=int, help="size of population")
    parser.add_argument("--n_iterations", type=int, help="number of iCEM iterations")
    parser.add_argument("--max_timesteps", type=int, help="maximum number of timesteps in environment")
    parser.add_argument("--planning_horizon", type=int, help="planning horizon")
    parser.add_argument("--pessimism_coef", type=float, default=None, help="pessimism coefficient")
    parser.add_argument("--threshold", type=float, default=None, help="threshold")
    parser.add_argument("--beta", type=float, help="beta coefficient for colored noise")
    parser.add_argument("--gamma", type=float, help="reward discount factor")
    parser.add_argument("--seed", type=int, default=123, help="seeds random, numpy, gym and pytorch modules")
    parser.add_argument("--model", type=str, default=None, help="path to dynamics model")
    parser.add_argument("--dataset", type=str, default=None, help="path to dataset")
    parser.add_argument("--bc_policy", type=str, default=None, help="path to BC policy")
    parser.add_argument("--q_fn", type=str, default=None, help="path to Q-function")
    parser.add_argument("--q_k", type=float, default=None, help="Q-value scaling")
    parser.add_argument(
        "--action_choice", type=str, choices=["mean", "first"], help="how the simulated actions are chosen"
    )
    parser.add_argument("--simulation", type=str, choices=["all", "individual"], help="how rollouts are simulated")
    parser.add_argument("--log_dir", type=str, default="logs", help="directory where log should be saved")
    parser.add_argument("--log_name", type=str, default="test", help="name of log")
    parser.add_argument("--alg", type=str, choices=[ALG_ICEM, ALG_MBOP, ALG_BC], help="iCEM or MBOP algorithm")
    args = parser.parse_args()
    if args.threshold == 0:
        args.threshold = np.inf

    main(args)
