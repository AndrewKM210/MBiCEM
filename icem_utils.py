import os
import pickle
import random
import sys
import time
from importlib import import_module

import numpy as np
import torch
import yaml
from dotmap import DotMap
from tabulate import tabulate

class Logger:
    def __init__(self, d=None):
        self.max_length = 5
        self.log = {}
        if d:
            self.update(d)

    def update(self, log):
        for k, v in log.items():
            if k in self.log.keys():
                if type(self.log[k]) is not list:
                    self.log[k] = [self.log[k]]
                if type(v) is list:
                    self.log[k] += v
                else:
                    self.log[k].append(v)
            else:
                self.log.update({k: v})

    def get_dict(self):
        return self.log

    def items(self):
        return self.log.items()

    def keys(self):
        return self.log.keys()

    def dump(self, log_name):
        print("Saving log to", log_name)
        pickle.dump(self, open(log_name, "wb"))

    def get(self, k):
        return self.log[k]

    def _print_dict(self, d, prefix=""):
        for k, v in d.items():
            print(f"{prefix}{k}:", end="")
            if type(v) is not list:
                print(f" {v}")
                continue
            if type(v) is list and len(v) > self.max_length:
                print(f" array({len(v)})")
                continue
            print()
            for vi in v:
                if type(vi) is dict:
                    self._print_dict(vi, prefix=prefix + "\t")
                if type(vi) is Logger:
                    self._print_dict(vi.log, prefix=prefix + "\t")
                else:
                    print(f"{prefix}\t{vi}")

    def print(self):
        self._print_dict(self.log)


def setup_log(dir, name):
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f'{name}_{time.strftime("%Y%m%d-%H%M%S")}.pickle')


def load_config_with_args(args):
    with open(args.config) as file:
        config = yaml.safe_load(file)

    with open(config["default_config"]) as file:
        default_config = yaml.safe_load(file)

    for k, v in config.items():
        default_config[k] = v

    args_dict = vars(args)
    for arg in args_dict:
        if args_dict[arg] is not None:
            default_config[arg] = args_dict[arg]

    return DotMap(default_config)


def set_seeds(seed):
    random.seed(seed)  # set python seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # sets torch seed for all devices (CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False


def print_config(config):
    stats = [[k, v] for k, v in config.items()]
    print("Parameters")
    print(tabulate(stats))
    print()


def load_reward_module(path):
    splits = path.split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]:
        dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    reward_module = import_module(filename)

    assert callable(getattr(reward_module, "reward_fn", None))
    assert callable(getattr(reward_module, "termination_fn", None))

    return reward_module


def print_exp_metrics(logs, episode=0):
    returns = [log.get("return") for log in logs]
    metrics = [[f"return_{episode}", int(returns[-1])]]
    if len(logs) > 1:
        metrics += [["returns_mean", int(np.mean(returns))], ["returns_std", int(np.std(returns))]]
    print(tabulate(metrics))
