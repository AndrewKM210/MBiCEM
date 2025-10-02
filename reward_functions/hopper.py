import numpy as np

# observaion mask for scaling
# 1.0 for positions and dt=0.02 for velocities
default_obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

def reward_fn(s, a, sp, obs_mask):
    sp = sp.clamp(-10.0, 10.0)
    a = a.clamp(-1.0, 1.0)

    vel_x = sp[..., -6]/obs_mask[-6] # forward reward
    height = sp[...,0]
    ang = sp[...,1]
    alive_bonus = 1.0 * (height > 0.7) * (ang.abs() <= 0.2)
    ctrl_cost = (a**2).sum(axis=-1) # ctrl_cost (power)
    r = vel_x + alive_bonus - 1e-3*ctrl_cost 
    # alive_bonus[alive_bonus == 0] = -100
    return r


def termination_fn(sp):
    return ~((sp[:,0] > 0.7) & (sp[:,1].abs() < 0.2) & (sp < 10).all(dim=-1))
    
