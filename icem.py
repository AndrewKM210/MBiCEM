from time import time
import colorednoise as c
import numpy as np
from icem_utils import Logger


class Icem:
    def __init__(self, params):
        # iCEM parameters
        self.pop_size = params.population_size
        self.n_iterations = params.n_iterations
        self.h = params.planning_horizon
        self.elite_size = params.elite_size
        self.elite_frac = params.elite_frac
        self.sigma_init = params.sigma_init
        self.beta = params.beta
        self.alpha = params.learning_rate
        self.gamma = params.gamma
        self.rng = np.random.default_rng(params.seed)

        # initialize iCEM variables
        self.low, self.high = params.min_max_action_value
        self.act_dim = params.act_dim
        self.mu_t = np.zeros((self.h, int(self.act_dim))) + (self.high + self.low) / 2
        self.sigma_t = np.full((self.h, int(self.act_dim)), (self.high - self.low) / 2 * params.sigma_init)
        self.elite_set = np.zeros((self.h, self.elite_size))
        self.elite_idx = np.zeros((self.elite_size))

    def get_action(self, s, t, model, **kwargs):
        if t > 0:  # shift mu
            self.mu_t[:-1, :] = self.mu_t[1:, :]
        self.sigma_t = np.full((self.h, int(self.act_dim)), (self.high - self.low) / 2 * self.sigma_init)
        log = Logger()
        for i in range(self.n_iterations):
            t_samples = time()
            samples = self._generate_samples(i, t)
            t_samples = time() - t_samples
            t_costs = time()
            costs, log_costs = model.calc_costs(samples, s)
            t_costs = time() - t_costs
            t_elites = time()
            self._update_elite_set(samples, costs[:, 0])
            t_elites = time() - t_elites
            t_params = time()
            self._update_params()
            t_params = time() - t_params
            log.update({"t_samples": t_samples, "t_costs": t_costs, "t_elites": t_elites, "t_params": t_params})
            log.update(log_costs)

        log.update({"r_pred": -costs[self.elite_idx[0], 1]})
        log.update({"h_pred": -costs[self.elite_idx[0], 0]})

        return np.copy(self.elite_set[0][0]), log, np.copy(self.elite_set[0])

    def _sample_colored_noise(self, num_samples):
        samples = c.powerlaw_psd_gaussian(self.beta, size=(num_samples, self.act_dim, self.h), random_state=self.rng)
        samples = samples.transpose([0, 2, 1])
        samples = np.clip(samples * self.sigma_t + self.mu_t, self.low, self.high)
        return samples

    def _generate_samples(self, i, t):
        if i == 0 and t == 0:  # no elites yet
            return self._sample_colored_noise(self.pop_size)

        num_samples = np.max((int(self.pop_size * np.power(self.gamma, -i)), 2 * self.elite_size))
        samples = self._sample_colored_noise(num_samples)
        keep_elite = int(self.elite_size * self.elite_frac)
        prev_elite = np.array(self.elite_set)
        if i == 0:  # shift previous elites
            prev_elite = prev_elite[:keep_elite, 1:, :]
            last_acts = self._sample_colored_noise(prev_elite.shape[0])[:, -1:None, :]
            prev_elite = np.concatenate([prev_elite, last_acts], axis=1)
        samples = np.concatenate((samples, prev_elite[:keep_elite]), axis=0)  # add fraction of previous elites

        # TODO: do always?
        if i == self.n_iterations - 1:  # add mean to samples
            samples[0] = self.mu_t

        return samples

    def _update_elite_set(self, samples, costs):
        self.elite_idx = costs.argsort()[: self.elite_size]
        self.elite_set = np.array([samples[i, :, :] for i in self.elite_idx])

    def _update_params(self):
        new_mu = self.elite_set.mean(axis=0)
        new_sigma = self.elite_set.std(axis=0)
        self.mu_t = np.array((1 - self.alpha) * new_mu + self.alpha * self.mu_t)
        self.sigma_t = np.array((1 - self.alpha) * new_sigma + self.alpha * self.sigma_t)
