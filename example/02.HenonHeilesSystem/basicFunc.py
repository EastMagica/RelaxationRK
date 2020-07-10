#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 20:46
# @file    : basicFunction.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np


def fun_fut(u, t):
    return np.array([
        u[0],
        u[1],
        u[2] + u[2] * u[3],
        u[3] + u[2] ** 2 - u[3] ** 2 * 2 / 3
    ])


def fun_ftu(t, u):
    return fun_fut(u, t)


def fun_h(u):
    # y = [p1, p2, q1, q2]
    return u[:2] ** 2 / 2 + u[2:] ** 2 / 2 + u[2] ** 2 * u[3] - u[2] ** 2 / 3


u01 = np.array([
    0.12, 0.12, 0.12, 0.12
])

u02 = np.array([
    0.12, 0.12, 0.12, np.sqrt(2 * 0.15925)
])


