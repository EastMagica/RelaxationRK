#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 18:57
# @file    : Relaxation.py
# @project : RelaxationRK
# software : PyCharm

import numpy as np
import matplotlib.pyplot as plt

from rkp.rrk.rrk4 import rk4_relax

from .basicFunc import fun_fut, u0, fun_he


# Example Functions
# ------------------

t0 = 0.0
h = 1e-1
step = 100


# RK Method
# ----------

fig, ax = plt.subplots()

y_array = rk4_relax(fun_fut, t0, u0, h, fun_he(y0), step)

ax.scatter(*y_array.T, s=5)

plt.show()
