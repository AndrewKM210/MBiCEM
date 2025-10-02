import numpy as np
import torch

scale = 0.02
default_obs_mask = np.array([1,1,1,1,1,1,1,1,scale,scale,scale,scale,scale,scale,scale,scale,scale])

def reward_fn(s, a, sp, obs_mask):

    sp = sp.clamp(-10.0, 10.0)
    a = a.clamp(-1.0, 1.0)

    # TODO: do everything with torch on GPU
    # forward_reward
    vel_x = (sp[..., 8]/obs_mask[8]).cpu().numpy() # forward reward
    
    # healthy reward
    height = sp[..., 0].cpu().numpy()
    ang = sp[..., 1].cpu().numpy()

    alive_bonus = 1.0 * np.logical_and(height > 0.8, height < 2.0) * (np.abs(ang) < 1.0)
    
    # ctrl_cost
    power = (a**2).sum(axis=-1).cpu().numpy()
    ctrl_cost = 1e-3*power
    
    r = vel_x + alive_bonus - ctrl_cost

    return torch.from_numpy(r).to(s.device)

def termination_fn(sp):
    return ~((sp[:,0] > 0.8) & (sp[:,0] < 2.0) & (sp[:,1].abs() < 1.0) & (sp < 10).all(dim=-1))

def reward_fn_gt(s, a, sp): 
    vel = s[8]
    act_cost = 1e-3 * (np.sum(a ** 2))
    r = vel - act_cost + 1

    return r      