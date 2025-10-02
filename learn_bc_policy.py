from datetime import datetime
import argparse
import pickle
import numpy as np
from bc_policy import Policy
import torch
import random
import os


def set_seeds(seed):
    random.seed(seed)  # set python seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # sets torch seed for all devices (CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False


def compute_ensemble_mse(policies, s, a, a_prior=None, batch_size=512):
    preds = []
    for p in policies:
        preds.append(p.predict_batched(s, a_prior))
    preds = np.mean(np.array(preds), axis=0)
    return np.mean(np.square(preds - a))


t_start = datetime.now().replace(microsecond=0)
parser = argparse.ArgumentParser(description="Learn an ensemble of policies from a dataset with BC")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--output", type=str, required=True, help="path to save ensemble")
parser.add_argument("--seed", type=int, default=123, help="seed for torch, numpy, etc.")
parser.add_argument("--ensemble_size", type=int, default=4, help="number of ensemble members")
parser.add_argument("--fit_epochs", type=int, default=300, help="number of training epochs")
parser.add_argument("--fit_lr", type=float, default=1e-3, help="fit learning rate")
parser.add_argument("--prior", action="store_true", help="use prior action for prediction")
parser.add_argument("--no_transform_in", action="store_true", help="do not apply transformations to input")
parser.add_argument("--holdout_ratio", type=float, default=0.0, help="holdout ratio of paths in dataset")
args = parser.parse_args()
set_seeds(args.seed)

print("BC policy learning parameters")
print("-----------------------------")
for key, value in vars(args).items():
    if key != "dataset" and key != "output":
        print(f"{key}: {value}")
print("-----------------------------")

print("\nLoading dataset", args.dataset)
d, metadata = pickle.load(open(args.dataset, "rb"))
d_h = None

if args.holdout_ratio > 0.0:
    rand_idx = np.random.permutation(len(d))
    paths_holdout = int(len(d) * args.holdout_ratio)
    d_h = [d[idx] for idx in rand_idx[:paths_holdout]]
    d = [d[idx] for idx in rand_idx[paths_holdout:]]

s_h, a_h, a_prior_h = None, None, None
if args.prior:
    s = np.concatenate([p["observations"][1:] for p in d])
    a = np.concatenate([p["actions"][1:] for p in d])
    a_prior = np.concatenate([p["actions"][:-1] for p in d])
    if d_h is not None:
        s_h = np.concatenate([p["observations"][1:] for p in d_h])
        a_h = np.concatenate([p["actions"][1:] for p in d_h])
        a_prior_h = np.concatenate([p["actions"][:-1] for p in d_h])
else:
    s = np.concatenate([p["observations"] for p in d])
    a = np.concatenate([p["actions"] for p in d])
    a_prior = None
    if d_h is not None:
        s_h = np.concatenate([p["observations"] for p in d_h])
        a_h = np.concatenate([p["actions"] for p in d_h])

print("samples_train:", s.shape[0])
if d_h is not None:
    print("samples_val:", s_h.shape[0])

policies = []
for i in range(args.ensemble_size):
    print("\n######################")
    print("Training BC policy", i)
    print("######################")
    policy = Policy(
        state_dim=s.shape[-1],
        act_dim=a.shape[-1],
        fit_lr=args.fit_lr,
        prior=args.prior,
        transform_in=not args.no_transform_in,
        seed=args.seed + i,
        id=i,
    )

    policy.fit(s, a, a_prior, s_h, a_h, a_prior_h, fit_mb_size=512, fit_epochs=args.fit_epochs)

    policies.append(policy)

print("\nDone training ensemble!")
print("\nCalculating ensemble MSE")
ensemble_mse = compute_ensemble_mse(policies, s, a, a_prior)
print("train_ensemble_mse:", ensemble_mse)
if args.holdout_ratio > 0:
    ensemble_mse = compute_ensemble_mse(policies, s_h, a_h, a_prior_h)
    print("val_ensemble_mse:", ensemble_mse)

print("\nSaving ensemble to", args.output)
pickle.dump(policies, open(args.output, "wb"))

t_exp = datetime.now().replace(microsecond=0) - t_start
print("\nExecution time:", t_exp)
print("Done!")
