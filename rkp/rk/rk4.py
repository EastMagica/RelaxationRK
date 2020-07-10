#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 14:14:20
# @file      : basic.py
# @project   : RelaxationRK
# @software  : VSCode

import numpy as np


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


def rk4_exp_cell(f, t0, y0, h, dim=1):
    k = np.zeros((4, dim))

    for j in range(4):
        k[j, :] = f(
            t0 + bt_rk4_exp["c"][j] * h,
            y0 + h * bt_rk4_exp["a"][j, :] @ k
        )
    return y0 + h * bt_rk4_exp["b"] @ k


def rk4_exp(f, t0, y0, h, step=1):
    n_dim = len(y0)
    y_array = np.zeros((step, n_dim))
    for i in range(step):
        y0 = rk4_exp_cell(f, t0, y0, h, n_dim)
        t0 += h
        y_array[i, :] = y0
    return y_array


