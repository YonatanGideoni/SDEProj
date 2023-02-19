import numpy as np
from matplotlib import pyplot as plt


def gen_ou(n_samples: int = 10 ** 3, mean: float = 1., var: float = 1., T: float = 1., beta: float = 1.) -> np.ndarray:
    init_point = np.random.randn() * var ** 0.5 + mean
    dt = T / n_samples
    incs = np.random.randn(n_samples - 1) * dt ** 0.5

    res = np.zeros(n_samples)
    res[0] = init_point
    for i in range(1, n_samples):
        res[i] = res[i - 1] - beta * res[i - 1] * dt + incs[i - 1] * var ** 0.5

    return res


def gen_pbm(n_samples: int = 10 ** 3, mean: float = 1., var: float = 1., T: float = 1.) -> np.ndarray:
    init_point = np.random.randn() * var ** 0.5 + mean
    dt = T / n_samples
    incs = np.random.randn(n_samples - 1) * dt ** 0.5

    res = np.zeros(n_samples)
    res[0] = init_point
    for i in range(1, n_samples):
        res[i] = res[i - 1] - res[i - 1] / (T - dt * i) * dt + incs[i - 1] * var ** 0.5

    return res


def calc_ou_score(process: np.ndarray, mean: float = 1, var: float = 1, T: float = 1., beta: float = 1) -> np.ndarray:
    t = np.linspace(0, T, process.shape[0])

    return -(process - np.exp(-beta * (T - t)) * mean) / ((var - 1) * np.exp(-2 * beta * (T - t)) + 1)


def calc_pbm_score(process: np.ndarray, mean: float = 1, var: float = 1, T: float = 1.) -> np.ndarray:
    t = np.linspace(0, T, process.shape[0])

    return -(process - t / T * mean) / (var * t / T * (T - t + t / T))


if __name__ == '__main__':
    for _ in range(25):
        ou = gen_ou()
        pbm = gen_pbm()

        ou_score = calc_ou_score(ou)
        pbm_score = calc_pbm_score(pbm)

        t = np.linspace(0, 1, ou_score.shape[0])

        plt.plot(t, ou_score, c='r', alpha=0.5)
        plt.plot(t, pbm_score, c='b', alpha=0.5)

    plt.ylim(-5, 5)

    plt.xlabel(r'$\frac{t}{T}$', fontsize=18)
    plt.ylabel(r'Score', fontsize=16)

    plt.show()
