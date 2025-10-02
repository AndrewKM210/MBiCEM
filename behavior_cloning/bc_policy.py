import torch
import numpy as np


class NN(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, activation_fn, seed, transform_in=False, prior=False):
        super(NN, self).__init__()

        torch.manual_seed(seed)

        self.prior = prior

        # nn layers dimensions
        self.state_dim, self.act_dim, self.hidden_size = state_dim, action_dim, hidden_size
        if self.prior:
            self.layer_sizes = (self.state_dim + self.act_dim,) + hidden_size + (self.act_dim,)
        else:
            self.layer_sizes = (self.state_dim,) + hidden_size + (self.act_dim,)

        # nn layers
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]) for i in range(len(self.layer_sizes) - 1)]
        )

        self.nonlinearity = activation_fn

        self.transform_in = transform_in
        self.s_shift, self.s_scale = None, None
        self.a_shift, self.a_scale = None, None

    def set_transformations(self, s, a=None):
        self.s_shift = np.mean(s, axis=0)
        self.s_scale = np.mean(np.abs(s - self.s_shift), axis=0)
        if self.prior:
            self.a_shift = np.mean(a, axis=0)
            self.a_scale = np.mean(np.abs(a - self.a_shift), axis=0)

    def transformations_to(self, device):
        self.s_shift = torch.from_numpy(self.s_shift).float().to(device)
        self.s_scale = torch.from_numpy(self.s_scale).float().to(device)
        if self.prior:
            self.a_shift = torch.from_numpy(self.a_shift).float().to(device)
            self.a_scale = torch.from_numpy(self.a_scale).float().to(device)

    def compute_decays(self):
        decay_0 = 1e-5 * (self.layers[0].weight ** 2).sum()
        decay_1 = 1e-5 * (self.layers[1].weight ** 2).sum()
        decay_2 = 1e-5 * (self.layers[2].weight ** 2).sum()
        factor = 1
        decays = (decay_0 + decay_1 + decay_2) * factor
        return decays

    def forward(self, s, a_prior=None):
        if self.prior:
            if a_prior is None:
                raise Exception("Missing prior action for prediction, prior action is None")
            if s.dim() != a_prior.dim():
                raise Exception("State and action inputs should be of the same size")
                exit(1)

        # normalize inputs
        s_in, a_in = s, a_prior
        if self.transform_in:
            s_in = (s - self.s_shift) / (self.s_scale + 1e-8)
            if self.prior:
                a_in = (a_prior - self.a_shift) / (self.a_scale + 1e-8)

        out = torch.cat([s_in, a_in], -1) if self.prior else s_in

        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.nonlinearity(out)
        out = self.layers[-1](out)
        return out


class Policy:
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size=(64, 64),
        seed=123,
        fit_lr=1e-3,
        device="cuda",
        transform_in=False,
        prior=False,
        scheduler="ExponentialLR",
        scheduler_gamma=0.99,
        id=0,
    ):
        self.id = id
        self.device = device
        self.prior = prior

        # nn
        self.state_dim, self.act_dim = state_dim, act_dim
        self.transform_in = transform_in
        activation_fn = torch.relu
        self.hidden_size = hidden_size
        self.out_dim = self.state_dim
        self.nn = NN(
            self.state_dim,
            self.act_dim,
            self.hidden_size,
            activation_fn,
            seed,
            transform_in=transform_in,
            prior=self.prior,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=fit_lr)
        self.mse_loss = torch.nn.MSELoss()

        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def to(self, device):
        self.nn.to(device)

    def fit(
        self,
        s,
        a,
        a_prior=None,
        s_h=None,
        a_h=None,
        a_prior_h=None,
        fit_mb_size=512,
        fit_epochs=5,
        gamma=0.99,
        track_mse=False,
    ):
        if self.transform_in:
            self.nn.set_transformations(s, a_prior)
            self.nn.transformations_to(self.device)

        n_samples = s.shape[0]
        n_batches = int(n_samples // fit_mb_size)
        fill_last_batch = False
        if n_samples != n_batches * fit_mb_size:
            n_batches += 1
            fill_last_batch = True
        train_loss = 0
        val_loss = None
        validate = not (s_h is None or a_h is None)

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
                if self.prior:
                    a_prior_batch = torch.from_numpy(a_prior[data_idx]).float().to(self.device)

                # predict
                if self.prior:
                    pred = self.nn(s_batch, a_prior_batch)
                else:
                    pred = self.nn(s_batch)

                # compute losses
                loss = self.mse_loss(pred, a_batch) + self.nn.compute_decays()

                # optimizer step and clip gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log metrics
                ep_loss += loss.detach().cpu().numpy()

            # update learning rate if using a scheduler
            if self.scheduler:
                self.scheduler.step()

            train_loss = ep_loss * 1.0 / n_batches

            if s_h is not None:
                train_loss = np.mean(np.square(self.predict_batched(s, a_prior) - a))
                val_loss = np.mean(np.square(self.predict_batched(s_h, a_prior_h) - a_h))

            print("\nEpoch", e)
            print(f"train_loss_{self.id}: {train_loss}")
            if validate:
                print(f"val_loss_{self.id}: {val_loss}")

    @torch.no_grad()
    def predict(self, s, a=None, to_cpu=True):
        if self.prior:
            assert type(s) is type(a)
            assert s.shape[0] == a.shape[0]
        if type(s) is np.ndarray:
            s = torch.from_numpy(s).float()
            if self.prior:
                a = torch.from_numpy(a).float()

        s = s.to(self.device)
        if self.prior:
            a = a.to(self.device)

        pred = self.nn.forward(s, a)
        pred = pred.detach().cpu().numpy() if to_cpu else pred
        return pred

    # Batch predict to lessen GPU usage
    @torch.no_grad()
    def predict_batched(self, s, a_prior=None, batch_size=256):
        if a_prior is not None:
            assert s.shape[0] == a_prior.shape[0]
        num_samples = s.shape[0]
        num_steps = int(num_samples // batch_size) + 1
        pred = np.ndarray((s.shape[0], self.act_dim))
        a_prior_batch = None

        for mb in range(num_steps):
            batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
            s_batch = s[batch_idx]
            if a_prior is not None:
                a_prior_batch = a_prior[batch_idx]
            pred_b = self.predict(s_batch, a_prior_batch)
            pred[batch_idx] = pred_b

        return pred
