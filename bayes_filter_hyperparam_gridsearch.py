import copy

import numpy as np
import pandas as pd
import torch

from consts import IMG_TENS_SHAPE, DEVICE, SIGMA, BS
from song_probnum_solver import solve_scipy, solve_magnani
from song_utils import marginal_prob_std


# TODO - move the generate_ground_truth function to some other generic file
# TODO - make sure that code changes you did here work and don't break anything in time_solvers.py
def generate_ground_truth(min_timestep: float, init_x: np.ndarray = None, tol: float = 1e-8) -> tuple:
    t = torch.ones(BS, device=DEVICE)

    if init_x is None:
        init_x = torch.randn(*IMG_TENS_SHAPE, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]
        init_x = np.concatenate([init_x.cpu().numpy().reshape((-1,)), np.zeros((IMG_TENS_SHAPE[0],))], axis=0)

    diffusion, ts = solve_scipy(copy.deepcopy(init_x), min_timestep, rtol=tol, atol=tol, method='RK45')
    gt = diffusion[:-1, -1]

    return gt, init_x


if __name__ == '__main__':
    sigmas = np.linspace(0.1, 3, num=20)
    thetas = np.linspace(-30, -0.1, num=20)
    qs = [1, 2, 3]

    min_timestep = 1e-7
    gt, init_x = generate_ground_truth(min_timestep)

    steps = 25
    final_time = 1e-7

    res = []
    for q in qs:
        for sigma in sigmas:
            hyp_params = {'sigma': sigma}

            res_over_time, _ = solve_magnani(copy.deepcopy(init_x), q=q, min_timestep=min_timestep, prior='IWP',
                                             steps=steps, solver_params=hyp_params)
            mse = ((res_over_time[-1][:-1, 0, 0] - gt) ** 2).mean()
            res.append(dict(prior='IWP', MSE=mse, diffusion=res_over_time, **hyp_params))

            for theta in thetas:
                hyp_params['theta'] = theta
                res_over_time, _ = solve_magnani(copy.deepcopy(init_x), q=q, min_timestep=min_timestep, prior='OU',
                                                 steps=steps, solver_params=hyp_params)
                mse = ((res_over_time[-1][:-1, 0, 0] - gt) ** 2).mean()
                res.append(dict(prior='IOU', MSE=mse, diffusion=res_over_time, **hyp_params))

            pd.DataFrame.from_records(res).to_csv('hyperparam_gridsearch_res.csv', index=False)

    res = pd.DataFrame.from_records(res)
    print(res.sort_values('MSE'))

    res.to_csv('hyperparam_gridsearch_res.csv', index=False)
