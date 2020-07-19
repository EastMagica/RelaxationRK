#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 21:01
# @file    : standard.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

from rkp.ssprk import ssprk33

from basicFunc import fun_f, u01, fun_he1, u02, fun_he2


# Example Functions
# ------------------

t0 = 0.0
h = 0.1
step = 500

u0 = u01
fun_he = fun_he1


# RK Method
# ----------


t_array, y_array = ssprk33(fun_f, t0, u0, h, step)

he_array = fun_he(y_array)


# Plot Solution
# --------------

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.8))

ax[0].scatter(y_array[:, 3], y_array[:, 1], s=5)
ax[1].plot(t_array, he_array, marker=".")

ax[0].set_xlim(-0.25, 0.25)
ax[0].set_ylim(-0.3, 0.3)

ax[1].set_xscale("symlog")
ax[1].set_yscale("symlog")
# ax[1].set_ylim(-1.5, 1e-13)

plt.tight_layout()
plt.show()
