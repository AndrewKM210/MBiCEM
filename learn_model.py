import os
import random
import numpy as np
import gym
import torch
import pickle
from probabilistic_dynamics_model  import ProbabilisticDynamicsEnsemble
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# set all seeds for reproducibility
def seed_torch(seed=123):
    random.seed(seed)  # set python seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # sets torch seed for all devices (CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # make cudnn deterministic

parser = argparse.ArgumentParser(description="iCEM with MOReL")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--output", type=str, default="test.pickle", help="output path for model")
args = parser.parse_args()

seed_torch(123)

if "halfcheetah" in args.dataset:
    env = "HalfCheetah-v3"
elif "hopper" in args.dataset:
    env = "Hopper-v3"
elif "walker" in args.dataset:
    env = "Walker2d-v3"
else:
    print("Error: unknown environmnt")
    exit(1)

e = gym.make(env)

data_path = args.dataset
# data_path = "datasets/d4rl_halfcheetah_medium_expert_v0_v02av02.pickle"
# data_path = "datasets/d4rl_hopper_medium_expert_v0_v02av02.pickle"


dataset, dataset_metadata = pickle.load(open(data_path, "rb"))

# dataset = random.sample(dataset, 100)

init_states_buffer = [p["observations"][0] for p in dataset]
s = np.concatenate([p["observations"][:-1] for p in dataset])
a = np.concatenate([p["actions"][:-1] for p in dataset])
sp = np.concatenate([p["observations"][1:] for p in dataset])
r = np.concatenate([p["rewards"][:-1] for p in dataset])
rollout_score = np.mean([np.sum(p["rewards"]) for p in dataset])
num_samples = np.sum([p["rewards"].shape[0] for p in dataset])

if "halfcheetah" in args.dataset:
    ensemble = ProbabilisticDynamicsEnsemble(e, 4, (200, 200, 200, 200), 123)
    fit_epochs = 500
else:
    ensemble = ProbabilisticDynamicsEnsemble(e, 4, (200, 200, 200, 200), 123)
    fit_epochs = 300

# ensemble = ProbabilisticDynamicsEnsemble(e, 1, (256, 256, 256), 123)
ensemble.fit_ensemble(s, a, sp, fit_epochs, 256, verbose=True)
pickle.dump(ensemble.models, open(args.output, "wb"))