from probnum import diffeq, filtsmooth, randvars, randprocs, problems
import numpy as np

import matplotlib.pyplot as plt
from probnum.diffeq.odefilter import ODEFilter
from probnum.diffeq.stepsize import ConstantSteps


def neg_exp_f(t, y):
    return -y


if __name__ == '__main__':
    t0 = 0
    tmax = 12.5
    y0 = np.array([1])
    ivp = problems.InitialValueProblem(t0=t0, tmax=tmax, f=neg_exp_f, y0=y0)

    iwp = randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=2,
        wiener_process_dimension=ivp.dimension,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    dt = 0.5
    steprule = ConstantSteps(dt)
    solver = ODEFilter(
        steprule=steprule,
        prior_process=iwp,
    )

    odesol = solver.solve(ivp)

    evalgrid = np.arange(ivp.t0, ivp.tmax, step=0.1)
    sol = odesol(evalgrid)

    plt.plot(evalgrid, sol.mean, "o-", linewidth=1)
    plt.xlabel("Time")
    plt.show()
