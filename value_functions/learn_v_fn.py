from datetime import datetime
import argparse
import pickle
import numpy as np
from v_fn import VFunction
from icem_utils import load_config_with_args
from dotmap import DotMap

Q_FN = "q_fn"
V_FN = "v_fn"

t_start = datetime.now().replace(microsecond=0)
parser = argparse.ArgumentParser(description="Learn an ensemble of value functions from a dataset")
parser.add_argument("--config", type=str, required=True, help="path to config")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--output", type=str, required=True, help="path to save ensemble")
parser.add_argument("--ensemble_size", type=int, help="number of ensemble members")
parser.add_argument("--fit_epochs", type=int, help="number of training epochs")
parser.add_argument("--fn", choices=[V_FN, Q_FN], help="learn V(s) or Q(s,a)")
args = parser.parse_args()
config = load_config_with_args(args)
config = DotMap(dict(((k, config[k]) for k in vars(args).keys())))

print("Value function learning parameters")
print("----------------------------------")
for key, value in config.items():
    if key != "dataset" and key != "output":
        print(f"{key}: {value}")
print("----------------------------------")

print("\nLoading dataset", config.dataset)
d, metadata = pickle.load(open(config.dataset, "rb"))

X = (
    np.concatenate([p["observations"][:-1] for p in d]),
    np.concatenate([p["actions"][:-1] for p in d]),
    np.concatenate([p["observations"][1:] for p in d]),
    np.concatenate([p["actions"][1:] for p in d]),
)
Y = [np.concatenate([p["rewards"][:-1] for p in d])]
print("n_samples:", Y[0].shape[0])

q_fn = config.fn == Q_FN
fns = []
for i in range(config.ensemble_size):
    print("\n##########################")
    print("Training value function", i)
    print("##########################")
    fn = VFunction(state_dim=X[0].shape[-1], act_dim=X[1].shape[-1], fit_lr=1e-3, seed=123 + i, q_fn=q_fn, id=i)
    fn.fit(X, Y, fit_mb_size=512, fit_epochs=config.fit_epochs)
    fns.append(fn)

    print("\nSaving ensemble to", config.output)
    pickle.dump(fns, open(config.output, "wb"))

t_exp = datetime.now().replace(microsecond=0) - t_start
print("\nExecution time:", t_exp)
print("Done!")
