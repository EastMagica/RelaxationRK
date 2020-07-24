#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 19:44
# @file    : basic.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np


u0 = np.array([1.0, 2.0])


def fun_f(t, u):
    return np.array([
        u[0] * (1 - u[1]),
        u[1] * (u[0] - 1)
    ])


def fun_h(u):
    return u[..., 0] + u[..., 1] - np.log(u[..., 0] * u[..., 1])


def fun_he(u):
    he0 = fun_h(u0)
    return he0 - fun_h(u)

