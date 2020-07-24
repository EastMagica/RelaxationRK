#!/usr/bin/python3
# -*- encoding:utf-8 -*-
# @author    : EastMagica
# @time      : 2020/07/18 22:59:18
# @file      : relaxation.py
# @project   : RelaxationRK
# @software  : VSCode

import matplotlib.pyplot as plt

from rkp.rrk.rrk import rk_relax

from basicFunc import fun_f, u01, fun_he1, u02, fun_he2


# Example Functions
# ------------------

t0 = 0.0
dt = 0.1
step = 500

u0 = u01
fun_he = fun_he1


# RK Method
# ----------

t_array, u_array, gamma_array = rk_relax(fun_f, t0, u0, dt, fun_he, step, order="3")

he_array = fun_he(u_array)


# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax[0].scatter(u_array[:, 2], u_array[:, 0], s=5)
ax[1].plot(t_array, he_array, marker=".")

# ax[1].set_xscale("symlog")
# ax[1].set_yscale("symlog")
# ax[1].set_ylim(-1, 1e-12)

plt.show()
