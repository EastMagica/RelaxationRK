#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/18 23:15:56
# @file      : ssprk33.py
# @project   : RelaxationRK
# @software  : VSCode


import numpy as np


# SSPRK(3,3) Methods
# -------------------

# Reference
# https://doi.org/10.1016/0021-9991(88)90177-5


# def ssprk33_cell(f, t0, u0, h, dim=1):
#     k1 = u0 + h * f(t0, u0)
#     k2 = 3/4 * u0 + 1/4 * k1 + 1/4 * h * f(t0+h, k1)
#     return 1/3 * u0 + 2/3 * k2 + 2/3 * h * f(t0+h/2, k2)


def ssprk33_cell(f, t0, u0, h, dim=1):
    k1 = u0 + h * f(t0, u0)
    k2 = u0 + h * (1/4 * f(t0, u0) + 1/4 * f(t0, k1))
    return u0 + h * (1/6 * f(t0, u0) + 1/6 * f(t0, k1) + 2/3 * f(t0, k2))


def ssprk33(f, t0, u0, h, step=1):
    n_dim = len(u0)
    t_array = np.arange(1, step+1) * h
    y_array = np.zeros((step, n_dim))
    for i in range(step):
        u0 = ssprk33_cell(f, t0, u0, h, n_dim)
        t0 += h
        y_array[i, :] = u0
    return t_array, y_array
