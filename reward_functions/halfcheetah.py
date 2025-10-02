import numpy as np
import math

scale = 0.02
default_obs_mask = np.array([1,1,1,1,1,1,1,1,scale,scale,scale,scale,scale,scale,scale,scale,scale])

def reward_fn(s, a, sp, obs_mask):
    sp = sp.clamp(-10.0, 10.0)
    a = a.clamp(-1.0, 1.0)

    vel_x = sp[..., 8]/obs_mask[8] # forward reward
    ang = sp[...,1]

    ctrl_cost = (a**2).sum(axis=-1) # ctrl_cost (power)
    r = vel_x - 0.1*ctrl_cost
    heading_penalty_factor = 10 # avoid rolling motion
    r -= (ang > math.pi/2) * heading_penalty_factor
    r -= (ang < -math.pi / 2) * heading_penalty_factor 
    return r

def termination_fn(sp):
    return ~((sp < 100).all(dim=-1))

def reward_fn_gt(s, a, sp): 
    if isinstance(sp, np.ndarray):
        vel = sp[8]
        ang = sp[1]
    else:
        vel = sp[2][0]
        ang = sp[1][2]
    
    act_cost = 0.1 * (np.sum(a ** 2))
    r = vel - act_cost
    heading_penalty_factor = 10
    r -= (ang > math.pi/2) * heading_penalty_factor
    r -= (ang < -math.pi / 2) * heading_penalty_factor    

    return r      

