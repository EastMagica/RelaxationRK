#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 13:54:21
# @file      : case.py
# @project   : RelaxationRK
# @software  : VSCode

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

from RRK.basic import rk4

# sys.path.append("C:\\CodeLibrary\\Python\\RelaxationRK\\RRK")

# Example Functions
# ------------------

def f(t, y):
    return np.array([
        y[0] * (1 - y[1]),
        y[1] * (y[0] - 1)
    ])

t0 = 0.0
y0 = np.array([1.0, 2.0])

h = 1e-3

step = 50000


# RK Method
# ----------


fig, ax = plt.subplots()

y_array = rk4(f, t0, y0, h, step)

ax.scatter(y_array[:, 0], y_array[:, 1], s=1)

plt.show()
