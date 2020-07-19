#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 20:46
# @file    : basicFunction.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np


u01 = np.array([
    0.12, 0.12, 0.12, 0.12
])

u02 = np.array([
    np.sqrt(2 * 0.15925), 0.12, 0.12, 0.12
])


def fun_f(t, u):
    return - np.array([
        - u[2] - 2 * u[2] * u[3],
        - u[3] - u[2] ** 2 + u[3] ** 2,
        u[0],
        u[1],
    ])


def fun_h(u):
    # y = [p1, p2, q1, q2]
    k1 = np.sum(u ** 2, axis=-1) / 2
    return k1 + u[..., 2] ** 2 * u[..., 3] - u[..., 3] ** 3 / 3


he01 = fun_h(u01)
he02 = fun_h(u02)


def fun_he1(u):
    return fun_h(u) - he01


def fun_he2(u):
    return fun_h(u) - he02
