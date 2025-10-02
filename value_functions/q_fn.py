import torch
import numpy as np


def swish(x):
    return x * torch.sigmoid(x)


class QNN(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, activation_fn, seed, transform_in=False):
        super(QNN, self).__init__()

        torch.manual_seed(seed)

        # nn layers dimensions
        self.state_dim, self.act_dim, self.hidden_size = state_dim, action_dim, hidden_size
        self.out_dim = 1  # mean, logvar
        self.layer_sizes = (self.state_dim + self.act_dim,) + hidden_size + (self.out_dim,)

        # nn layers
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        )

        self.nonlinearity = torch.relu

        self.transform_in = transform_in
        self.s_shift, self.s_scale = None, None
        self.a_shift, self.a_scale = None, None

    def set_transformations(self, s, a, device):
        self.s_shift = np.mean(s, axis=0)
        self.a_shift = np.mean(a, axis=0)
        self.s_scale = np.mean(np.abs(s - self.s_shift), axis=0)
        self.a_scale = np.mean(np.abs(a - self.a_shift), axis=0)

    def transformations_to(self, device):
        self.s_shift = torch.from_numpy(self.s_shift).float().to(device)
        self.a_shift = torch.from_numpy(self.a_shift).float().to(device)
        self.s_scale = torch.from_numpy(self.s_scale).float().to(device)
        self.a_scale = torch.from_numpy(self.a_scale).float().to(device)

    def compute_decays(self):
        decay_0 = 1e-5 * (self.layers[0].weight ** 2).sum()
        decay_1 = 1e-5 * (self.layers[1].weight ** 2).sum()
        decay_2 = 1e-5 * (self.layers[2].weight ** 2).sum()
        factor = 1
        decays = (decay_0 + decay_1 + decay_2) * factor
        return decays

    def forward(self, s, a, ret_logvar=False):
        if s.dim() != a.dim():
            print("Error: State and action inputs should be of the same size")
            exit(1)

        # normalize inputs
        if self.transform_in:
            s_in = (s - self.s_shift) / (self.s_scale + 1e-8)
            a_in = (a - self.a_shift) / (self.a_scale + 1e-8)
        else:
            s_in, a_in = s, a

        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.nonlinearity(out)
        out = self.layers[-1](out)
        return out


class QFunction:
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size=(512, 512),
        seed=123,
        fit_lr=5e-4,
        fit_wd=0.0,
        device="cuda",
        activation="relu",
        transform_in=False,
        scheduler="ExponentialLR",
        scheduler_gamma=0.995,
        id=0,
    ):
        self.id = id
        self.device = device

        # nn
        self.state_dim, self.act_dim = state_dim, act_dim
        self.transform_in = transform_in
        activation_fn = swish if activation == "swish" else torch.relu
        self.hidden_size = hidden_size
        self.out_dim = self.state_dim
        self.q_fn = QNN(
            self.state_dim, self.act_dim, self.hidden_size, activation_fn, seed, transform_in=transform_in
        ).to(self.device)
        self.q_target = QNN(
            self.state_dim, self.act_dim, self.hidden_size, activation_fn, seed, transform_in=transform_in
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_fn.parameters(), lr=fit_lr)
        self.mse_loss = torch.nn.MSELoss()

        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def to(self, device):
        self.q_fn.to(device)
        self.q_target.to(device)

    def fit(
        self,
        X,
        Y,
        fit_mb_size,
        fit_epochs,
        gamma=0.99,
        max_steps=1e4,
        track_mse=False,
        *args,
        **kwargs,
    ):
        s, a, sp, ap = X
        r = Y[0]

        if self.transform_in:
            self.q_fn.set_transformations(s, a, self.device)
            self.q_fn.transformations_to(self.device)
            self.q_target.set_transformations(s, a, self.device)
            self.q_target.transformations_to(self.device)

        n_samples = s.shape[0]
        n_batches = int(n_samples // fit_mb_size)
        fill_last_batch = False
        if n_samples != n_batches * fit_mb_size:
            n_batches += 1
            fill_last_batch = True

        for e in range(fit_epochs):
            rand_idx = np.random.permutation(n_samples)
            ep_loss = 0.0

            for b in range(n_batches):
                # get batch of data, fill with random samples if last batch
                data_idx = rand_idx[b * fit_mb_size : (b + 1) * fit_mb_size]
                if b == n_batches - 1 and fill_last_batch:
                    fill_size = (fit_mb_size - data_idx.shape[0],)
                    fill_idx = rand_idx[: b * fit_mb_size][np.random.randint(0, b * fit_mb_size, fill_size)]
                    data_idx = np.concatenate([data_idx, fill_idx])

                # move batch to GPU
                s_batch = torch.from_numpy(s[data_idx]).float().to(self.device)
                a_batch = torch.from_numpy(a[data_idx]).float().to(self.device)
                sp_batch = torch.from_numpy(sp[data_idx]).float().to(self.device)
                ap_batch = torch.from_numpy(ap[data_idx]).float().to(self.device)
                r_batch = torch.from_numpy(r[data_idx]).float().to(self.device).unsqueeze(-1)

                # predict
                pred = self.q_fn(s_batch, a_batch)
                target = r_batch + gamma * self.q_target(sp_batch, ap_batch)

                # compute losses
                loss = self.mse_loss(pred, target) + self.q_fn.compute_decays()

                # optimizer step and clip gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log metrics
                ep_loss += loss.detach().cpu().numpy()

            # update learning rate if using a scheduler
            if self.scheduler:
                self.scheduler.step()

            ep_loss = ep_loss * 1.0 / n_batches

            print("\nEpoch", e)
            print(f"loss_{self.id}: {ep_loss}")

            # update target
            q_fn_params = dict(self.q_fn.named_parameters())
            q_target_params = dict(self.q_target.named_parameters())
            tau = 0.005
            for k, v in self.q_target.named_parameters():
                v.data = (1 - tau) * q_target_params[k].data + tau * q_fn_params[k].data

    def forward(self, s, a):
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) is np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.q_fn.forward(s, a)

    def predict(self, s, a, to_cpu=True):
        assert type(s) is type(a)
        assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
        s, a = s.to(self.device), a.to(self.device)
        pred = self.q_fn.forward(s, a)
        pred = pred.detach().cpu().numpy() if to_cpu else pred
        return pred

    def predict_batched(self, s, a, batch_size=256, to_cpu=True):
        assert type(s) is type(a)
        assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()

        # Batch predict to lessen GPU usage
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size) + 1
        pred = np.ndarray((s.shape))

        for mb in range(num_steps):
            batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
            s_batch = s[batch_idx].to(self.device)
            a_batch = a[batch_idx].to(self.device)
            pred_b = self.q_fn.forward(s_batch, a_batch)

            pred_b = pred_b.to("cpu").data.numpy()
            pred[batch_idx] = pred_b
        return pred
