#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author  : east
# @time    : 2020/7/10 21:01
# @file    : standard.py
# @project : RelaxationRK
# software : PyCharm

import matplotlib.pyplot as plt

from rkp.rk import rk4

from basicFunc import fun_ftu, u01


# Example Functions
# ------------------

h = 0.001
step = 1000


# RK Method
# ----------


y_array = rk4(fun_ftu, 0.0, u01, h, step)

fig, ax = plt.subplots()
ax.scatter(y_array.T[3], y_array.T[1], s=5)

plt.show()
