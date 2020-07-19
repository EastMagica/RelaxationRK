#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 13:54:21
# @file      : case.py
# @project   : RelaxationRK
# @software  : VSCode

import matplotlib.pyplot as plt

from rkp.rk import rk4

from basicFunc import fun_f, fun_he, u0


# Example Functions
# ------------------

t0 = 0.0
h = 0.85
step = 20


# RK Method
# ----------

t_array, y_array = rk4(fun_f, t0, u0, h, step)

he_array = fun_he(y_array)


# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax[0].scatter(*y_array.T, s=5)
ax[1].plot(t_array, he_array, marker=".")

ax[1].set_xscale("symlog")
ax[1].set_yscale("symlog")
ax[1].set_ylim(-1.5, 1e-13)

plt.show()
