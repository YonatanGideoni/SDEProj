import os
import pickle

import numpy as np
from matplotlib import pyplot as plt, gridspec


def load_res(res_folder: str, partial_res_name: str):
    for file_name in os.listdir(res_folder):
        if partial_res_name in file_name:
            path = os.path.join(res_folder, file_name)
            return pickle.load(open(path, 'rb'))


def load_cached_res(res_folder: str, partial_res_name: str):
    res = load_res(res_folder, partial_res_name)

    # 7 corresponds to 50 steps
    return res['diffusions'][7][:, -1].reshape(28, 28)


def plot_cached_err_over_time(folder, gt_name, finite_name, ax, n_pixels_plot: int = 10, c='r'):
    cached_res = load_res(folder, gt_name)
    gt_trajs = cached_res['diffusions'][-1][:n_pixels_plot]
    gt_times = np.linspace(1, 0, num=gt_trajs.shape[-1])

    cached_res = load_res(folder, finite_name)
    finite_trajs = cached_res['diffusions'][7][:n_pixels_plot]
    times = np.linspace(1, 0, num=finite_trajs.shape[-1])

    for gt_traj, traj in zip(gt_trajs, finite_trajs):
        ax.plot(times, traj - np.interp(times[::-1], gt_times[::-1], gt_traj.numpy()[::-1])[::-1], c=c)


def plot_cached_traj(folder, file_name, ax, n_pixels_plot: int = 10, c='k'):
    cached_res = load_res(folder, file_name)
    trajs = cached_res['diffusions'][7][:n_pixels_plot]
    times = np.linspace(1, 0, num=trajs.shape[-1])

    for traj in trajs:
        ax.plot(times, traj, c=c)


if __name__ == '__main__':
    fs = 20
    res_folder = 'normal_bf_res'

    fig = plt.figure()

    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    iwp1_ax = fig.add_subplot(spec[0, 0])
    iwp2_ax = fig.add_subplot(spec[1, 0])
    iwp1 = load_cached_res(res_folder, 'iwp1')
    iwp2 = load_cached_res(res_folder, 'iwp2')
    iwp1_ax.imshow(iwp1)
    iwp2_ax.imshow(iwp2)
    iwp1_ax.set_title('IWP, q=1', fontsize=fs)
    iwp2_ax.set_title('IWP, q=2', fontsize=fs)

    ou1_ax = fig.add_subplot(spec[0, 1])
    ou2_ax = fig.add_subplot(spec[1, 1])
    ou1 = load_cached_res(res_folder, 'ou1')
    ou2 = load_cached_res(res_folder, 'ou2')
    ou1_ax.imshow(ou1)
    ou2_ax.imshow(ou2)
    ou1_ax.set_title('IOUP, q=1', fontsize=fs)
    ou2_ax.set_title('IOUP, q=2', fontsize=fs)

    euler_ax = fig.add_subplot(spec[0, 2])
    gt_ax = fig.add_subplot(spec[1, 2])
    euler = load_cached_res(res_folder, 'euler')
    scipy = load_cached_res(res_folder, 'sci45')
    euler_ax.imshow(euler)
    euler_ax.set_title('Euler', fontsize=fs)

    gt = load_res(res_folder, 'gt')['ms'][:-1, -1].reshape(28, 28)
    gt_ax.imshow(gt)
    gt_ax.set_title('Ground truth', fontsize=fs)

    axs = [iwp1_ax, iwp2_ax, ou1_ax, ou2_ax, euler_ax, gt_ax]
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])

    plt.subplots_adjust(wspace=0.)

    # TODO - finish this
    fig, axs = plt.subplots(3, 1)
    plot_cached_traj('normal_bf_res', 'iwp1', axs[0])
    plot_cached_traj('const_var_res', 'iwp1', axs[1])
    # plot_cached_traj('semi_int_res', 'euler', axs[2])

    plt.show()
