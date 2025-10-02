import numpy as np

av_scale = 0.02
v_scale = 0.02

default_obs_mask = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., v_scale, v_scale, v_scale, av_scale,
       av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, av_scale, 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1.])

def reward_fn(paths, obs_mask=None):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)
    
    # TODO: add clip check
    # obs = np.clip(paths["observations"], -100.0, 100.0)
    obs = np.clip(paths["next_observations"], -10.0, 10.0)
    act = paths["actions"].clip(-1.0, 1.0)
    
    # forward_reward
    if obs.shape[-1] == 28:
        vel_x = obs[:,:,-1]/obs_mask[26]
    elif obs.shape[-1] == 112:
        vel_x = obs[:,:,27]/obs_mask[26]
    else:
        vel_x = obs[:, :, 13]/obs_mask[13]
    
    # healthy reward
    z_coordinate = obs[:, :, 0]

    alive_value = 1
    alive_bonus = alive_value * np.logical_and(z_coordinate > 0.2, z_coordinate < 1)
    
    
    # ctrl_cost
    power = np.square(act).sum(axis=-1) 
    ctrl_cost = power

    # contact cost
    # contact_forces = np.clip(obs[:,:,27:], a_min=-1,a_max=1)
    # contact_cost = np.sum(np.square(contact_forces))
    # https://arxiv.org/pdf/1807.03858.pdf
    
    # rewards = vel_x + alive_bonus - 0.5*ctrl_cost - 5e-4 *contact_cost
    rewards = vel_x + alive_bonus - 0.5*ctrl_cost

    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths

def termination_fn(paths):
    for path in paths:
        obs = path["observations"]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            done = not (
                np.isfinite(np.abs(obs[t])).all()
                and (np.abs(obs[t]) < 10).all()
                and (obs[t][0] > 0.2)
                and (obs[t][0] < 1)
            )
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths

