#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 13:54:21
# @file      : case.py
# @project   : RelaxationRK
# @software  : VSCode

import matplotlib.pyplot as plt

from rkp.rk import rk4

from basicFunc import fun_fut, u0


# Example Functions
# ------------------

t0 = 0.0
h = 0.085
step = 500


# RK Method
# ----------

y_array = rk4(fun_fut, t0, u0, h, step)

fig, ax = plt.subplots()
ax.scatter(*y_array.T, s=5)

plt.show()
