from datetime import datetime
import argparse
import pickle
import numpy as np
from q_fn import QFunction


t_start = datetime.now().replace(microsecond=0)
parser = argparse.ArgumentParser(description="Learn an ensemble of Q-functions from a dataset")
parser.add_argument("--dataset", type=str, required=True, help="path to dataset")
parser.add_argument("--output", type=str, required=True, help="path to save ensemble")
parser.add_argument("--ensemble_size", type=int, default=4, help="number of ensemble members")
parser.add_argument("--fit_epochs", type=int, default=300, help="number of training epochs")
args = parser.parse_args()
print("Loading dataset", args.dataset)
d, metadata = pickle.load(open(args.dataset, "rb"))

X = (
    np.concatenate([p["observations"][:-1] for p in d]),
    np.concatenate([p["actions"][:-1] for p in d]),
    np.concatenate([p["observations"][1:] for p in d]),
    np.concatenate([p["actions"][1:] for p in d]),
)
Y = [np.concatenate([p["rewards"][:-1] for p in d])]
print("Number of samples in dataset:", Y[0].shape[0])

q_fns = []
for i in range(args.ensemble_size):
    print("\n######################")
    print("Training Q-function", i)
    print("######################")
    q_fn = QFunction(state_dim=X[0].shape[-1], act_dim=X[1].shape[-1], fit_lr=1e-3, seed=123 + i, id=i)
    q_fn.fit(X, Y, fit_mb_size=512, fit_epochs=args.fit_epochs)
    q_fns.append(q_fn)

print("\nSaving ensemble to", args.output)
pickle.dump(q_fns, open(args.output, "wb"))

t_exp = datetime.now().replace(microsecond=0) - t_start
print("\nExecution time:", t_exp)
print("Done!")
