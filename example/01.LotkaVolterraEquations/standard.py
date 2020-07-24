#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/10 13:54:21
# @file      : case.py
# @project   : RelaxationRK
# @software  : VSCode

import matplotlib.pyplot as plt

from rkp.rk import rk_exp

from basicFunc import fun_f, fun_he, u0


# Example Functions
# ------------------

t0 = 0.0
dt = 0.1
step = 500


# RK Method
# ----------

t_array, u_array = rk_exp(fun_f, t0, u0, dt, step)

he_array = fun_he(u_array)

print(u_array)
print(he_array)

# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax[0].scatter(*u_array.T, s=5)
ax[1].plot(t_array, he_array, marker=".")

ax[1].set_xscale("symlog")
ax[1].set_yscale("symlog")
# ax[1].set_ylim(-1.5, 1e-13)

plt.show()
