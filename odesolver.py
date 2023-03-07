import numpy as np
import scipy
import jax.numpy as jnp
import jax

# Implement algorithm  for 1 dimension, and vmap over dimensions
h = 0.1
q = 1
sigma = 1.0

def factorial(n):
    return jnp.round(jnp.exp(jax.scipy.special.gammaln(n + 1)), 0)

def get_A(q, h):
    js = jnp.arange(q + 1)
    ij = js - js[..., None]
    ij = ij  * (ij >= 0)

    num = h ** ij
    denom = factorial(ij)

    return jnp.triu(num / denom)

def get_Q(q, h, sigma=0.1):
    js = jnp.arange(q + 1)
    ipj = (2 * q + 1) - (js + js[...,None])
    qmj = jax.scipy.special.gammaln(q - js + 1)

    qmj = jnp.round(jnp.exp(qmj), 0)
    qfaci = qmj * qmj[...,None]

    num = h ** ipj
    denom = qfaci * ipj

    return sigma ** 2 * num / denom


def get_A_OU(q, h, theta):
    A = get_A(q, h)

    js = jnp.arange(q + 1)
    q_minus_is = q - js

    num = jnp.exp(theta * h)

    def _calc_num_i(i):
        num = jnp.exp(theta * h)

        for k in range(q - i): # So that it only goes from k=0 to k=q-i-1
            num -= ((theta * h) ** (k)) / factorial(k)
        
        return num / (theta) ** (q - i)

    col_q = jnp.array([_calc_num_i(i) for i in jnp.arange(q + 1)])

    A = A.at[:, q].set(col_q)

    return A


def get_Q_OU(q, h, theta, sigma=1.):
    
    js = jnp.arange(q + 1)
    ipj = (2 * q) - (js + js[...,None])

    print(ipj)  # 2q - i - j

    scaling_factor = sigma ** 2 / (theta ** ipj)  # sigma^2 / theta^(2q - i - j)

    first_term = (jnp.exp(2. * theta * h) - 1) / (2. * theta)

    # Calculating the k term
    def _calc_sum_k_term(i):
        # Calculate individual term
        def _calc_k_term(k):
            term = ((-1) ** k) * (jnp.exp(theta * h) - 1) / theta
            for l in range(1, k + 1): # So that it only goes from l=1 to l=k
                term += ((-1) ** (k - l)) * (theta) ** (l - 1) * jnp.exp(theta * h) * h ** l / factorial(l)

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

    k1_k2_terms = jnp.array([[_calc_sum_k1_k2_term(i, j) for i in jnp.arange(q + 1)] for j in jnp.arange(q + 1)])

    k_term_with_i = jnp.array([_calc_sum_k_term(i) for i in jnp.arange(q + 1)])

    k_term = k_term_with_i + k_term_with_i[..., None]
    
    first_term -= k_term
    first_term += k1_k2_terms

    Q = scaling_factor * first_term

    return Q

from functools import partial

def solve_kf(m0, P0, f : callable, t0=0, t1=8, steps=50, R=0.0, q=2, method='OU', theta=1., sigma=1.):

    m, P = m0, P0

    all_ms = [m0]
    
    h = -(t1 - t0) / steps
    ts = jnp.linspace(float(t0), float(t1), steps + 1)

    print(f'Using t0={t0}, t1={t1} for {steps} steps')
    print(f'ts={ts}')
    print(f'Using h={h} for {steps} steps')

    if method == 'OU':
        A = get_A_OU(q, h, theta=theta)
        Q = get_Q_OU(q, h, theta=theta, sigma=sigma)
    else:
        A = get_A(q, h)
        Q = get_Q(q, h, sigma=sigma)
    
    @partial(jax.vmap, in_axes=(0, 0, None))
    def kalman_predict(m, P, t_i):
        m_minus = A @ m

        P_minus = A @ P @ A.T + Q
        return m_minus, P_minus
    
    @partial(jax.vmap, in_axes=(0, 0, 0, None))
    def kalman_step(z, m_minus, P_minus, t_i):
        v = z  - m_minus[1, :]

        S = P_minus[1, 1] + R

        K = P_minus[:, [1]] * (1.0 / S)
        
        # euler-like method
        # make z have the same shape as v so this works
        # breakpoint()
        # m = m_minus + K @ z.reshape((1,))[None]*h

        # P = P_minus

        m = m_minus + K @ v[None]

        P = P_minus - K @ S[None, None] @ K.T
        return m, P
    
    for t_i in ts[:-1]:
        
        # Predict
        m_minus, P_minus = kalman_predict(m, P, t_i)
        z = f(t_i, m_minus[:, 0, :].T).T # This odd transpose is because we assume
                                         # the vector fields second axis is dimension of the state space
    
        # Update
        # euler-like step
        # m, P = kalman_step(z, m, P, t_i)


        m, P = kalman_step(z, m_minus, P_minus, t_i)
        
        all_ms.append(m_minus)

    return all_ms, ts

if __name__ == "__main__":

    t1 = 8
    h = 0.5
    steps = int(t1 / h)

    q = 2
    m0 = jnp.array([1., 1.0])
    P0 = jnp.array([
        [0, 0],
        [0, 1],
    ])


    m0 = jnp.zeros(q+1).at[:2].set(m0)
    P0 = jnp.eye(q+1).at[0,0].set(0.0)

    # quick 2D test

    m0 = jnp.array([m0, m0])[..., None]
    P0 = jnp.array([P0, P0])


    def g(t, x):
        scaling = jnp.array([1, 1.06])[None, ...]
        return x * scaling

    print(m0.shape, P0.shape)
    ms, ts = solve_kf(m0, P0, g, t1=t1, steps=steps, q=q)

    import matplotlib.pyplot as plt

    ms = jnp.array(ms)
    print(ms.shape, P0.shape)
    plt.plot(ts, ms[:, 0, 0])
    plt.plot(ts, jnp.exp(ts))
    plt.ylim(top=2500)
    plt.show()
    # plt.plot(ts, ms[:, 1, 0])
    # plt.plot(ts, ms[:, 1, 0])