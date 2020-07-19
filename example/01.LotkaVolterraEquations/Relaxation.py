#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 18:57
# @file    : Relaxation.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

from rkp.rrk.rrk4 import rk4_relax

from basicFunc import fun_f, u0, fun_he


# Example Functions
# ------------------

t0 = 0.0
h = 0.85
step = 200


# RK Method
# ----------

t_array, y_array = rk4_relax(fun_f, t0, u0, h, fun_he, step)

he_array = fun_he(y_array)


# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax[0].scatter(*y_array.T, s=5)
ax[1].plot(t_array, he_array)

ax[1].set_yscale("symlog")
# ax[1].set_ylim(-1, 1e-12)

plt.show()
