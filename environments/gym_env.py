import gym
import numpy as np
import pickle
from dotmap import DotMap
import torch

class GymEnv:
    def __init__(self, env_name, reward_fn, obs_mask, v_fns=None, seed=123):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.reward_fn = reward_fn
        self.seed = seed
        self.obs_mask = obs_mask
        self.v_fns = v_fns

    def get_observation_dimension(self):
        return self.env.observation_space.shape[0]
    
    def get_action_dimension(self):
        return self.env.action_space.shape[0]

    def get_min_max_action_value(self):
        return self.env.action_space.low, self.env.action_space.high

    def reset(self):
        return self.env.reset()

    def set_state(self, state):
        self.env.reset()
        self.env.sim.set_state(state)
        self.env.sim.forward()

    def set_state_from_obs(self, qpos, qvel):
        self.env.reset()
        self.env.sim.set_state(qpos, qvel)
        self.env.sim.forward()

    def get_state(self):
        return self.env.sim.get_state()

    def simulate_sequence(self, action_seq, gamma=0.99):
        reward_sum = 0
        step_cost = 0
        gamma = 0.99
        state = self.get_state()
        s = state
        sp = None
        for i, a in enumerate(action_seq):
            sp, r, done, _ = self.env.step(a)
            if self.reward_fn is not None and type(self.reward_fn) is not DotMap:
                r = self.reward_fn(s, a, sp)
            reward_sum += r * (gamma**i)
            if i == 0:
                step_cost = -r
            if done:
                if "hopper" in self.env_name.lower():
                    reward_sum -= 100
                break
            s = sp
        self.set_state(state)
        cost = -reward_sum
        max_err = 0
        if sp is not None and self.v_fns is not None:
            s = torch.from_numpy(np.array([sp*self.obs_mask])).float().to("cuda")
            a = torch.from_numpy(np.array([action_seq[-1]])).float().to("cuda")
            values = []
            for v_fn in self.v_fns:
                if v_fn.q_fn:
                    values.append(v_fn.predict(s, a)[0][0])
                else:
                    values.append(v_fn.predict(s)[0][0])
            max_err = 0
            for i in range(4):
                for j in range(4):
                    if j > i:
                        max_err = max(max_err, np.abs(values[i]-values[j]))
            # if max_err < 50:
            v = np.mean(values) * (gamma**action_seq.shape[0])
            cost -= v
            # cost -= self.q_fns.predict(s, a)[0][0]
        return cost, step_cost, max_err
    
    def simulate_sequence_policy(self, s, noise, policy, q_fn, low=-1, high=1, gamma=0.99):
        reward_sum = 0
        step_cost = 0
        state = self.get_state()
        sp = None
        seq = []
        env_done = False
        for i, n in enumerate(noise):
            a = np.clip(policy(s*self.obs_mask).detach().cpu().numpy() + n, low, high)
            seq.append(a)
            sp, r, done, _ = self.env.step(a)
            if self.reward_fn is not None and type(self.reward_fn) is not DotMap:
                r = self.reward_fn(s, a, sp)
            if not env_done:
                reward_sum += r * (gamma**i)
            if i == 0:
                step_cost = -r
            if done:
                if "hopper" in self.env_name.lower() and not env_done:
                    reward_sum -= 100
                env_done=True
            s = sp
        self.set_state(state)
        cost = -reward_sum
        if q_fn is not None and sp is not None:
            ap = policy(s*self.obs_mask).detach().cpu().numpy()
            v = q_fn[0].predict(sp[np.newaxis,...], ap[np.newaxis,...])[0][0]
            cost -= v
        return cost, step_cost, np.array(seq)

    def get_next_states(self, action_seq):
        next_states = []
        step_costs = []
        for i in range(action_seq.shape[0]):
            state = self.get_state()
            sp, r, done, _ = self.env.step(action_seq[i, 0])
            self.set_state(state)
            next_states.append(sp)
            step_costs.append(-r)
        return np.array(next_states), np.array(step_costs)

    def calc_costs(self, samples, s):
        ret = []
        discs = []
        for seq in samples:
            c, s_c, disc = self.simulate_sequence(seq, gamma=1)
            ret.append((c, s_c))
            discs.append(disc)
        return np.array(ret), {"v": 0}
    
    def calc_costs_policy(self, noises, s, policy, q_fn, low=-1, high=1):
        costs = []
        samples = []
        for noise in noises:
            cost, step_cost, seq = self.simulate_sequence_policy(s, noise, policy, q_fn, low, high, gamma=1)
            costs.append([cost, step_cost])
            samples.append(seq)
        costs = np.array(costs)
        samples = np.array(samples)

        if False:
            kappa = 0.3
            exp_cost = np.exp(costs[:,0]*kappa)
            exp_cost_sum = np.sum(exp_cost)
            weighed_samples = []
            for t in range(samples.shape[1]): # for timestep t
                weighed_sample = 0
                for n in range(samples.shape[0]): # for trajectory n
                    weighed_sample += np.exp(costs[n,0]*kappa) * samples[n,t]
                weighed_sample /= exp_cost_sum
                weighed_samples.append(weighed_sample)
            weighed_samples = np.array(weighed_samples)
            return costs,  {"v": 0}, weighed_samples
        else:
            ordered_idx = costs[:,0].argsort()
            samples = samples[ordered_idx][0]
            return costs,  {"v": 0}, samples

    def execute_step(self, action):
        s, r, done, info = self.env.step(action)
        cost = -r
        return done, cost, s, info

    def render_env(self):
        return self.env.render()
