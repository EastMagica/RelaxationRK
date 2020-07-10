#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 18:56
# @file    : rrk.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np

from scipy.optimize import root


# Explicit RK Methods
# --------------------

bt_rk4_exp = {
    "a": np.array([
        [0,   0,   0, 0],
        [1/2, 0,   0, 0],
        [0,   1/2, 0, 0],
        [0,   0,   0, 1]
    ], dtype=np.float64),
    "b": np.array([
        1/6, 1/3, 1/3, 1/6
    ], dtype=np.float64),
    "c": np.array([
        0, 1/2, 1/2, 1
    ], dtype=np.float64)
}


def rk4_relax_cell(f, t0, y0, h, gamma, dim=1):
    k = np.zeros((4, dim))

    for j in range(4):
        k[j, :] = f(
            t0 + bt_rk4_exp["c"][j] * h,
            y0 + h * bt_rk4_exp["a"][j, :] @ k
        )
    return y0 + gamma * h * bt_rk4_exp["b"] @ k


def rk4_relax(f, t0, y0, h, fun_h, step=1):
    n_dim = len(y0)
    y_array = np.zeros((step, n_dim))
    for i in range(step):
        sol = root(
            lambda gam: fun_h(rk4_relax_cell(f, t0, y0, h, gam, n_dim)),
            x0=np.zeros(1, dtype=np.float),
            method="hybr",
            options={
                "xtol": 1e-14
            }
        )
        print("sol:\n", sol)
        gamma = sol.x[0]
        y0 = rk4_relax_cell(f, t0, y0, h, gamma, n_dim)
        print("-" * 64)
        print("y0:", y0)
        print("-" * 64)
        t0 += gamma * h
        y_array[i, :] = y0
    return y_array