import numpy as np

def reward_fn(paths, target):

    obs = paths["observations"].clip(-100.0, 100.0)
    paths["next_observations"].clip(-100.0, 100.0)
    paths["actions"].clip(-1.0, 1.0)
    
    rewards = (np.linalg.norm(obs[:,:,0:2] - target, axis=-1) <= 0.5).astype(float)
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths


def termination_fn(paths):
    for path in paths:
        obs = path["observations"]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            done = not (np.isfinite(np.abs(obs[t])).all())
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths
