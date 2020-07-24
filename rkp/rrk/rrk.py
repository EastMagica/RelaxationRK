#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 18:56
# @file    : rrk.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np
from scipy.optimize import root

from rkp.rk.explicit import rk_exp_cell


# Relax Explicit RK Methods
# ---------------------------


def rk_relax(f, t0, u0, dt, fun_he, step=1, order="4"):
    n_dim = len(u0)
    t_array = np.zeros(step)
    u_array = np.zeros((step, n_dim))
    gamma_array = np.ones(step)
    for i in range(step):
        sol = root(
            lambda gam: fun_he(u0 + gam * rk_exp_cell(f, t0, u0, dt, n_dim, order)),
            x0=np.ones(1, dtype=np.float64),
            method="hybr",
            options={
                "xtol": 1e-14
            }
        )
        gamma = sol.x[0]
        u0 += gamma * rk_exp_cell(f, t0, u0, dt, n_dim, order)
        t0 += gamma * dt
        t_array[i] = t0
        u_array[i, :] = u0
        gamma_array[i] = gamma
    return t_array, u_array, gamma_array
