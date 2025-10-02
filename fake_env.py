from time import time

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm


def merge_tensors(a, b, idx_a, idx_b, shape, device):
    full = torch.zeros(shape).to(device)
    if len(a) > 0:
        full[idx_a] = a
    if len(b) > 0:
        full[idx_b] = b
    return full


class FakeEnv:
    def __init__(self, model, reward_fn, termination_fn, params, q_fn, bc_policy, k=1):
        # Learned dynamics variables
        self.models = model
        self.n_models = len(model)
        self.model = model[0]
        self.use_mean = params.use_mean
        self.n_nets = 4

        # P-MDP variables
        self.thresh = None
        self.pen = params.penalty
        self.simulate_all = params.simulation != "individual"

        # Reward and termination functions
        self.r_fn = reward_fn
        self.t_fn = termination_fn
        self.gamma = params.gamma

        # Dataset variables
        self.obs_mask = None
        self.act_repeat = 1
        self.include_velocity = False

        # GPU
        self.device = self.model.device

        # Q-function and BC policy
        self.q_fn = q_fn
        self.bc_policy = bc_policy
        self.k = k

    def setup(self, dataset, pc, metadata, thresh=None):
        # Update settings
        self.obs_mask = metadata["obs_mask"]
        if "act_repeat" in metadata.keys():
            self.act_repeat = metadata["act_repeat"]
        else:
            self.act_repeat = 1
        if "include_velocity" in metadata.keys():
            self.include_velocity = metadata["include_velocity"]

        # Get elite members
        s = np.concatenate([p["observations"][:-1] for p in dataset])
        a = np.concatenate([p["actions"][:-1] for p in dataset])
        sp = np.concatenate([p["observations"][1:] for p in dataset])
        if self.n_models > self.n_nets and hasattr(self.model, "holdout_idx"):
            print(f"\nSelecting the best {self.n_nets} MLPs out of {self.n_models}")
            val_losses = []
            holdout_idx = self.model.holdout_idx
            s_val = s[holdout_idx]
            a_val = a[holdout_idx]
            sp_val = sp[holdout_idx]
            s = np.delete(s, holdout_idx, axis=0)
            a = np.delete(a, holdout_idx, axis=0)
            sp = np.delete(sp, holdout_idx, axis=0)
            for m in tqdm(self.models):
                val_losses.append(m.compute_loss_batched(s_val, a_val, sp_val))
            best_idx = np.argsort(np.array(val_losses))
            self.models = [self.models[i] for i in best_idx[: self.n_nets]]
        elif self.n_models > self.n_nets:
            self.models = self.models[: self.n_nets]
        self.n_models = len(self.models)

        # Compute threshold
        if pc == "None" or pc == 0:
            print("Using naive MBRL")
            return
        if thresh:
            print("\nWarning: threshold set manually!")
            self.thresh = thresh
            print("threshold\t", self.thresh)
            return

        print("\nComputing disagreement betweeen members of ensemble")
        delta = np.zeros(s.shape[0])

        with tqdm(total=6) as pbar:
            for idx_1, model_1 in enumerate(self.models):
                pred_1 = model_1.predict_batched(s, a)
                for idx_2, model_2 in enumerate(self.models):
                    if idx_2 > idx_1:
                        pred_2 = model_2.predict_batched(s, a)
                        disagreement = np.linalg.norm((pred_1 - pred_2), axis=-1)
                        delta = np.maximum(delta, disagreement)
                        pbar.update(1)
        self.thresh = np.mean(delta) + pc * np.std(delta)

        print(
            tabulate(
                [
                    ["threshold", self.thresh],
                    ["disc_mean", np.mean(delta)],
                    ["disc_std", np.std(delta)],
                    ["disc_max\t", np.max(delta)],
                ]
            )
        )

    def compute_disc_all(self, preds):
        delta = torch.zeros(preds[0].shape[0]).to(self.device)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[0]):
                if j > i:
                    disc = (preds[i] - preds[j]).norm(dim=-1)
                    delta = torch.max(delta, disc)
        return delta

    def compute_disc(self, preds):
        delta = torch.zeros(preds[0].shape[0]).to(self.device)
        for i in range(self.n_models):
            for j in range(self.n_models):
                if j > i:
                    disc = (preds[i] - preds[j]).norm(dim=-1)
                    delta = torch.max(delta, disc)
        return delta

    @torch.no_grad()  # save GPU memory by disabling gradient calculation
    def simulate_sequences_icem(self, acts, s, gamma=0.99):
        if self.obs_mask is not None:
            s *= self.obs_mask

        # send data to GPU
        pen = torch.Tensor([self.pen]).to(self.device)
        s = torch.Tensor(s).expand((acts.shape[0], s.shape[0])).to(self.device).float()  # (pop_size, obs_dim)
        if self.simulate_all:  # TODO: repeat or expand?
            s = s.repeat(self.n_models, 1, 1)  # (n_models, pop_size, obs_dim)
        costs_halt = torch.zeros(0).to(self.device).float()
        idx_halt = torch.zeros(0).to(self.device).long()
        idx = torch.arange(0, acts.shape[0]).to(self.device)
        acts = torch.from_numpy(acts).to(self.device).float()  # (pop_size, h, act_dim)
        costs = torch.zeros(acts.shape[0]).to(self.device)  # (pop_size)
        step_costs = None
        costs_shape = costs.shape
        for i in range(acts.shape[1]):  # iterate through horizon
            # predict
            t_f = time()
            preds = []
            for j in range(self.n_models):
                if self.simulate_all:
                    pred = self.models[j].forward(s[j], acts[:, i])
                else:
                    pred = self.models[j].forward(s, acts[:, i])
                preds.append(pred[0] if type(pred) is tuple else pred)
            t_f = time() - t_f

            # compute disagreement
            # TODO: fix
            if self.thresh is None:
                self.thresh = np.inf
            t_d = time()
            disc = self.compute_disc(preds)
            t_d = time() - t_d

            # find indices with unknown states
            t_p = time()
            if self.simulate_all:
                v_idx = torch.where(disc > self.thresh * 2)[0]
                nv_idx = torch.where(disc <= self.thresh * 2)[0]
            else:
                v_idx = torch.where(disc > self.thresh)[0]
                nv_idx = torch.where(disc <= self.thresh)[0]

            # penalize and add unknown states to halt
            costs_halt = torch.cat([costs_halt, costs[v_idx] + pen])
            idx_halt = torch.cat([idx_halt, idx[v_idx]])

            # keep known states
            idx = idx[nv_idx]
            costs = costs[nv_idx]
            acts = acts[nv_idx]
            preds = torch.stack(preds)[:, nv_idx]
            sp = preds.mean(dim=0)
            if self.simulate_all:
                s = s.mean(dim=0)
            s = s[nv_idx]
            t_p = time() - t_p

            # compute costs
            t_r = time()
            c = -self.r_fn(s, acts[:, i], sp, self.obs_mask) * gamma**i
            costs += c

            # termination function
            done = self.t_fn(sp)
            costs_halt = torch.cat([costs_halt, costs[done]+pen])
            idx_halt = torch.cat([idx_halt, idx[done]])
            idx = idx[~done]
            costs = costs[~done]
            acts = acts[~done]
            preds = preds[:, ~done]
            sp = sp[~done]

            # record predicted first step costs
            if i == 0:
                step_costs = merge_tensors(costs, costs_halt, idx, idx_halt, costs_shape, self.device)
                step_costs = step_costs.detach().cpu().numpy()
            t_r = time() - t_r

            s = preds if self.simulate_all else sp

            # stop if all paths have been truncated
            if s.numel() == 0:
                break

            if self.q_fn is not None and self.q_fn[0].q_fn and i == acts.shape[1]-2:
                break

        if self.q_fn is not None and sp.numel() > 0:
            sp = sp * torch.from_numpy(self.obs_mask).float().to(self.device)
            ap = acts[:, -1]  # TODO: wrong!
            preds = []
            for q_fn in self.q_fn:
                # preds.append(self.q_fn[0].predict(sp[np.newaxis, ...], ap[np.newaxis, ...], to_cpu=False)[0,:,0])
                if q_fn.q_fn:
                    preds.append(q_fn.predict(sp[np.newaxis, ...], ap[np.newaxis, ...], to_cpu=False)[0,:,0])
                else:
                    preds.append(q_fn.predict(sp[np.newaxis, ...], to_cpu=False)[0,:,0])
            v = torch.mean(torch.stack(preds), dim=0)*self.k
            # print(v.mean(), v.max(), v.min()) # in the order of -0.2, '0.5', -1.13
            costs -= v

        t_e = time()
        full_costs = torch.zeros(costs_shape).to(self.device)
        if len(costs) > 0:
            full_costs[idx] = costs
        n_v = 0
        if len(costs_halt) > 0:
            full_costs[idx_halt] = costs_halt
            n_v = costs_halt.shape[0]
        n_v /= full_costs.shape[0]
        costs = full_costs.detach().cpu().numpy()
        costs = np.array([(costs[i], step_costs[i]) for i in range(costs.shape[0])])

        t_e = time() - t_e
        log = {"v": n_v, "t_c_f": t_f, "t_c_r": t_r, "t_c_d": t_d, "t_c_p": t_p, "t_c_e": t_e}
        return costs, log

    @torch.no_grad()  # save GPU memory by disabling gradient calculation
    def simulate_sequences_mbop(self, acts, s, low, high, gamma=0.99):
        if self.obs_mask is not None:
            s *= self.obs_mask

        # send data to GPU
        pen = torch.Tensor([self.pen]).to(self.device)
        s = torch.Tensor(s).expand((acts.shape[0], s.shape[0])).to(self.device).float()  # (pop_size, obs_dim)
        costs_halt = torch.zeros(0).to(self.device).float()
        idx_halt = torch.zeros(0).to(self.device).long()
        idx = torch.arange(0, acts.shape[0]).to(self.device)
        acts = torch.from_numpy(acts).to(self.device).float()  # (pop_size, h, act_dim)
        costs = torch.zeros(acts.shape[0]).to(self.device)  # (pop_size)
        step_costs = None
        costs_shape = costs.shape
        a = torch.zeros((acts.shape[0], acts.shape[2])).to(self.device).float()

        for i in range(acts.shape[1]):  # iterate through horizon
            # predict
            t_f = time()
            a = torch.clamp(self.bc_policy.get_action_gpu(s, a, False)[0] + acts[:, i], low[0], high[0])
            if i == 0:
                first_a = a.detach().cpu().numpy()
            preds = []
            for j in range(self.n_models):
                pred = self.models[j].forward(s, a)
                preds.append(pred[0] if type(pred) is tuple else pred)
            t_f = time() - t_f

            # compute disagreement
            # TODO: fix
            if self.thresh is None:
                self.thresh = np.inf

            t_d = time()
            disc = self.compute_disc(preds)
            t_d = time() - t_d

            # find indices with unknown states
            t_p = time()
            v_idx = torch.where(disc > self.thresh)[0]
            nv_idx = torch.where(disc <= self.thresh)[0]

            # penalize and add unknown states to halt
            costs_halt = torch.cat([costs_halt, costs[v_idx] + pen])
            idx_halt = torch.cat([idx_halt, idx[v_idx]])

            # keep known states
            idx = idx[nv_idx]
            costs = costs[nv_idx]
            acts = acts[nv_idx]
            a = a[nv_idx]
            preds = torch.stack(preds)[:, nv_idx]

            sp = preds.mean(dim=0)
            s = s[nv_idx]
            t_p = time() - t_p

            # compute costs
            t_r = time()
            if s.numel() > 0:
                c = -self.r_fn(s, a, sp, self.obs_mask) * gamma**i
                costs += c

            # termination function
            # idxs_keep, idxs_done = self.t_fn(sp)
            # # print(idxs_done.shape)
            # costs_halt = torch.cat([costs_halt, costs[idxs_done]+pen])
            # idx_halt = torch.cat([idx_halt, idx[idxs_done]])
            # idx = idx[idxs_keep]
            # costs = costs[idxs_keep]
            # acts = acts[idxs_keep]
            # preds = preds[:, idxs_keep]
            # sp = sp[idxs_keep]

            # record predicted first step costs
            if i == 0:
                step_costs = merge_tensors(costs, costs_halt, idx, idx_halt, costs_shape, self.device)
                step_costs = step_costs.detach().cpu().numpy()
            t_r = time() - t_r

            s = sp

            # stop if all paths have been truncated
            if s.numel() == 0:
                break

        if self.q_fn is not None and sp.numel() > 0:
            sp = sp * torch.from_numpy(self.obs_mask).float().to(self.device)
            ap = self.policy(sp)
            v = self.q_fn[0].predict(sp[np.newaxis, ...], ap[np.newaxis, ...], to_cpu=False)[0][0]
            costs -= v

        t_e = time()
        full_costs = torch.zeros(costs_shape).to(self.device)
        if len(costs) > 0:
            full_costs[idx] = costs
        n_v = 0
        if len(costs_halt) > 0:
            full_costs[idx_halt] = costs_halt
            n_v = costs_halt.shape[0]
        n_v /= full_costs.shape[0]
        costs = full_costs.detach().cpu().numpy()

        costs = np.array([(costs[i], step_costs[i]) for i in range(costs.shape[0])])
        ordered_idx = costs[:, 0].argsort()
        samples = first_a[ordered_idx]

        t_e = time() - t_e
        log = {"v": n_v, "t_c_f": t_f, "t_c_r": t_r, "t_c_d": t_d, "t_c_p": t_p, "t_c_e": t_e}
        return costs, log, samples

    def calc_costs(self, samples, s):
        init_state = s.copy()
        return self.simulate_sequences_icem(samples, init_state, gamma=self.gamma)

    def calc_costs_mbop(self, samples, s, low, high):
        init_state = s.copy()
        return self.simulate_sequences_mbop(samples, init_state, low, high, gamma=self.gamma)
