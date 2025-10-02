import numpy as np
from icem_utils import Logger
from tqdm import tqdm


class Agent:
    def __init__(self, params):
        self.seed = params.seed
        self.t_max = params.max_timesteps

    def rollout(self, env, policy, model=None, seed=123, no_tqdm=False):
        env.env.seed(seed)
        s = env.reset()
        log = Logger()
        obs = [s]
        acts = []
        rewards = []
        reward_sum = 0
        model = env if not model else model
        pbar = tqdm(range(self.t_max), disable=no_tqdm)
        a = np.zeros(env.get_action_dimension())
        for t in pbar:
            a, log_new, a_seq = policy.get_action(s=s, t=t, model=model, a_prior=a)
            a = np.clip(a, -1, 1)
            acts.append(a)
            sp, r, done, info = env.env.step(a)
            obs.append(sp)
            rewards.append(r)
            reward_sum += r
            s = sp
            pbar.set_postfix({"ret": reward_sum, "r": r})
            if done:
                break
        log.update({"obs": obs, "acts": acts, "rewards": rewards, "return": reward_sum})
        return log
