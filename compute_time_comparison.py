import numpy as np
import torch
from timeit import default_timer as timer

from consts import DEVICE
from song_utils import img_tens_shape, score_model, BS, SIGMA, marginal_prob_std


def time_model_pass(sample: np.ndarray, time_steps: np.ndarray) -> float:
    assert img_tens_shape[0] == 1, 'Faithfully measuring only a single (non-batched) forward pass'

    sample = torch.tensor(sample, device=DEVICE, dtype=torch.float32).reshape(img_tens_shape)
    time_steps = torch.tensor(time_steps, device=DEVICE, dtype=torch.float32).reshape((sample.shape[0],))
    with torch.no_grad():
        start_time = timer()
        score_model(sample, time_steps)
        tot_runtime = timer() - start_time

    return tot_runtime


def time_rand_mat_inv(mat_size: int) -> float:
    mat = torch.randn((mat_size, mat_size), device=DEVICE)

    start_time = timer()

    torch.linalg.inv(mat)  # not optimal as in practice we'd use `linalg.solve` but good enough for OOM estimate

    tot_runtime = timer() - start_time

    return tot_runtime


if __name__ == '__main__':
    model_pass_runtimes = []
    rand_mat_inv_runtimes = []

    n_runs = 10 ** 4

    t = torch.ones(BS, device=DEVICE)

    for _ in range(n_runs):
        sample = torch.randn(*img_tens_shape, device=DEVICE) * marginal_prob_std(t, SIGMA)[:, None, None, None]

        pass_time = time_model_pass(sample, t)
        mat_inv_time = time_rand_mat_inv(img_tens_shape[-1])

        model_pass_runtimes.append(pass_time)
        rand_mat_inv_runtimes.append(mat_inv_time)

    print(f'Model pass time:{np.mean(model_pass_runtimes):.2e}+-{np.std(model_pass_runtimes):.2e}\n'
          f'Mat inv time: {np.mean(rand_mat_inv_runtimes):.2e}+-{np.std(rand_mat_inv_runtimes):.2e}')
