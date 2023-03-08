from functools import partial

import numpy as np
import torch
from functorch import vmap

# Implement algorithm  for 1 dimension, and vmap over dimensions
h = 0.1
q = 1
sigma = 1.0


def factorial(n):
    return torch.round(torch.exp(torch.special.gammaln(torch.tensor(n + 1))), decimals=0)


def get_A(q, h):
    js = torch.arange(q + 1)
    ij = js - js[..., None]
    ij = ij * (ij >= 0)

    num = h ** ij
    denom = factorial(ij)

    return torch.triu(num / denom)


def get_Q(q, h, sigma=0.1):
    js = torch.arange(q + 1)
    ipj = (2 * q + 1) - (js + js[..., None])
    qmj = torch.special.gammaln(q - js + 1)

    qmj = torch.round(torch.exp(qmj), decimals=0)
    qfaci = qmj * qmj[..., None]

    num = h ** ipj
    denom = qfaci * ipj

    return sigma ** 2 * num / denom


def get_A_OU(q, h, theta):
    A = get_A(q, h)

    def _calc_num_i(i):
        num = torch.exp(theta * h)

        for k in range(q - i):  # So that it only goes from k=0 to k=q-i-1
            num -= ((theta * h) ** (k)) / factorial(k)

        return num / (theta) ** (q - i)

    col_q = torch.Tensor([_calc_num_i(i) for i in torch.arange(q + 1)])

    A[:, q] = col_q

    return A


def get_Q_OU(q, h, theta, sigma=1.):
    js = torch.arange(q + 1)
    ipj = (2 * q) - (js + js[..., None])

    scaling_factor = sigma ** 2 / (theta ** ipj)  # sigma^2 / theta^(2q - i - j)

    first_term = (torch.exp(2. * theta * h) - 1) / (2. * theta)

    # Calculating the k term
    def _calc_sum_k_term(i):
        # Calculate individual term
        def _calc_k_term(k):
            term = ((-1) ** k) * (torch.exp(theta * h) - 1) / theta
            for l in range(1, k + 1):  # So that it only goes from l=1 to l=k
                term += ((-1) ** (k - l)) * (theta) ** (l - 1) * torch.exp(theta * h) * h ** l / factorial(l)

            return term

        # Calculate the sum over k
        sum_term = 0

        for k in range(q - i):
            sum_term += _calc_k_term(k)

        return sum_term

    # Calculating the k1_k2_term at the end
    def _calc_sum_k1_k2_term(i, j):
        # Calculate individual term
        def _calc_k1_k2_term(k1, k2):
            term = theta ** (k1 + k2) * h ** (k1 + k2 + 1)

            term /= ((k1 + k2 + 1) * factorial(k1) * factorial(k2))

            return term

        # Calculate the sum over k1 and k2
        sum_term = 0

        for k1 in range(q - i):
            for k2 in range(q - j):
                sum_term += _calc_k1_k2_term(k1, k2)

        return sum_term

    k1_k2_terms = torch.Tensor([[_calc_sum_k1_k2_term(i, j) for i in torch.arange(q + 1)] for j in torch.arange(q + 1)])

    k_term_with_i = torch.Tensor([_calc_sum_k_term(i) for i in torch.arange(q + 1)])

    k_term = k_term_with_i + k_term_with_i[..., None]

    # can't do -= because of pytorch broadcasting
    first_term = first_term - k_term
    first_term += k1_k2_terms

    Q = scaling_factor * first_term

    return Q


def solve_kf(m0, P0, f: callable, t0=0, t1=8, steps=50, R=0.0, q=2, method='OU', theta=1., sigma=1.):
    m, P = m0, P0

    all_ms = [m0]

    h = torch.tensor(-(t1 - t0) / steps)
    ts = torch.linspace(float(t0), float(t1), steps + 1)

    print(f'Using t0={t0}, t1={t1} for {steps} steps')
    print(f'ts={ts}')
    print(f'Using h={h} for {steps} steps')

    if method == 'OU':
        A = get_A_OU(q, h, theta=theta)
        Q = get_Q_OU(q, h, theta=theta, sigma=sigma)
    else:
        A = get_A(q, h)
        Q = get_Q(q, h, sigma=sigma)

    @partial(vmap, in_dims=(0, 0, None))
    def kalman_predict(m, P, t_i):
        m_minus = A @ m

        P_minus = A @ P @ A.T + Q
        return m_minus, P_minus

    @partial(vmap, in_dims=(0, 0, 0, None))
    def kalman_step(z, m_minus, P_minus, t_i):
        v = z - m_minus[1, :]

        S = P_minus[1, 1] + R

        K = P_minus[:, [1]] * (1.0 / S)

        m = m_minus + K @ v[None]

        P = P_minus - K @ S[None, None] @ K.T
        return m, P

    for t_i in ts[:-1]:
        # Predict
        m_minus, P_minus = kalman_predict(m, P, t_i)
        z = f(t_i, m_minus[:, 0, :].T).T  # This odd transpose is because we assume
        # the vector fields second axis is dimension of the state space

        # Update
        m, P = kalman_step(z, m_minus, P_minus, t_i)

        all_ms.append(m_minus)

    return all_ms, ts


if __name__ == "__main__":
    t1 = 8
    h = 0.5
    steps = int(t1 / h)

    q = 2
    m0 = torch.Tensor([1., 1.0])
    P0 = torch.Tensor([
        [0, 0],
        [0, 1],
    ])

    # TODO - torchify
    m0 = torch.zeros(q + 1).at[:2].set(m0)
    P0 = torch.eye(q + 1).at[0, 0].set(0.0)

    # quick 2D test

    m0 = torch.Tensor([m0, m0])[..., None]
    P0 = torch.Tensor([P0, P0])


    def g(t, x):
        scaling = torch.Tensor([1, 1.06])[None, ...]
        return x * scaling


    print(m0.shape, P0.shape)
    ms, ts = solve_kf(m0, P0, g, t1=t1, steps=steps, q=q)

    import matplotlib.pyplot as plt

    ms = torch.Tensor(ms)
    print(ms.shape, P0.shape)
    plt.plot(ts, ms[:, 0, 0])
    plt.plot(ts, torch.exp(ts))
    plt.ylim(top=2500)
    plt.show()
    # plt.plot(ts, ms[:, 1, 0])
    # plt.plot(ts, ms[:, 1, 0])
