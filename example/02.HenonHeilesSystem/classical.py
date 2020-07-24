#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 20:54
# @file    : classical.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

from scipy.integrate import odeint, solve_ivp

from basicFunc import fun_ftu, u01


# Example Functions
# ------------------

dt = 0.1
step = 100


# RK Method
# ----------


# sol = odeint(fun_f, y0, np.arange(step+1)*h)
sol = solve_ivp(
    fun=fun_ftu, t_span=(0.0, 10), y0=u01, method="RK45"
)

fig, ax = plt.subplots()
ax.scatter(sol.y[3], sol.y[1], s=5)

plt.show()
