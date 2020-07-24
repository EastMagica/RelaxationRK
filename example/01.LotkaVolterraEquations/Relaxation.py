#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 18:57
# @file    : Relaxation.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

from rkp.rrk import rk_relax

from basicFunc import fun_f, u0, fun_he


# Example Functions
# ------------------

t0 = 0.0
dt = 0.85
step = 200


# RK Method
# ----------

t_array, u_array, gamma_array = rk_relax(fun_f, t0, u0, dt, fun_he, step)

he_array = fun_he(u_array)


# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax[0].scatter(*u_array.T, s=5)
ax[1].plot(t_array, he_array)

ax[1].set_yscale("symlog")
# ax[1].set_ylim(-1, 1e-12)

plt.show()
