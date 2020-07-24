#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 19:30
# @file    : classical.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

# odeint (old), solve_ivp (new)
from scipy.integrate import solve_ivp

from basicFunc import fun_f, u0


# Example Functions
# ------------------

dt = 0.85
step = 100


# RK Method
# ----------


# sol = odeint(fun_f, y0, np.arange(step+1)*h)
sol = solve_ivp(
    fun=fun_f, t_span=(0.0, 85), y0=u0, method="RK45"
)

fig, ax = plt.subplots()
ax.scatter(*sol.y, s=5)

plt.show()
