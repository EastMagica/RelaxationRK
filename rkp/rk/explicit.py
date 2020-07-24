#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 14:14:20
# @file      : basic.py
# @project   : RelaxationRK
# @software  : VSCode


r"""
Explicit RK Methods

"""

import numpy as np


# Butcher tableau
# ----------------

bt_exp_dict = {
    "4": {
        "k": 4,
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
    },
    "ssp3": {
        "k": 3,
        "a": np.array([
            [0,     0,   0],
            [1,     0,   0],
            [1/4, 1/4,   0],
        ], dtype=np.float64),
        "b": np.array([
            1/6, 1/6, 2/3
        ], dtype=np.float64),
        "c": np.array([
            0, 1, 1/2
        ], dtype=np.float64)
    },
}


# Explicit RK Method
# -------------------

def rk_exp_cell(f, t0, u0, dt, n_dim=None, order="4"):
    n_dim = np.size(u0) if n_dim is None else n_dim
    bt_temp = bt_exp_dict[order]
    k = np.zeros((bt_temp["k"], n_dim))

    for i in range(bt_temp["k"]):
        k[i, :] = f(
            t0 + bt_temp["c"][i] * dt,
            u0 + dt * bt_temp["a"][i, :] @ k
        )
    return dt * bt_temp["b"] @ k


def rk_exp(f, t0, u0, dt, step=1, order="4"):
    n_dim = np.size(u0)
    t_array = np.arange(1, step+1) * dt
    u_array = np.zeros((step, n_dim))
    for i in range(step):
        u0 += rk_exp_cell(f, t0, u0, dt, n_dim, order)
        t0 += dt
        u_array[i, :] = u0
    return t_array, u_array
