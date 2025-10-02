from time import time
import colorednoise as c
import numpy as np
from icem_utils import Logger


class MBOP:
    def __init__(self, params):
        # iCEM parameters
        self.pop_size = params.population_size
        self.h = params.planning_horizon
        self.sigma_init = params.sigma_init
        self.beta = params.beta
        self.gamma = params.gamma
        self.rng = np.random.default_rng(params.seed)
        self.low, self.high = params.min_max_action_value
        self.act_dim = params.act_dim
        self.sigma_t = np.full((self.h, int(self.act_dim)), (self.high - self.low) / 2 * params.sigma_init)

    def get_action(self, s, t, model, **kwargs):
        log = Logger()
        t_samples = time()
        noise = self._generate_noise()
        t_samples = time() - t_samples
        t_costs = time()
        costs, log_costs, samples = model.calc_costs_mbop(noise, s, self.low, self.high)
        t_costs = time() - t_costs
        t_elites = time()
        t_elites = time() - t_elites
        t_params = time()
        t_params = time() - t_params
        log.update({"t_samples": t_samples, "t_costs": t_costs, "t_elites": t_elites, "t_params": t_params})
        log.update(log_costs)
        return samples[0], log, samples

    def _generate_noise(self):
        noise = c.powerlaw_psd_gaussian(self.beta, size=(self.pop_size, self.act_dim, self.h), random_state=self.rng)
        noise[0] = np.zeros_like(noise[0])
        return np.clip(noise.transpose([0, 2, 1]) * self.sigma_t, -0.05, 0.05)
