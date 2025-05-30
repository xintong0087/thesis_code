import numpy as np
import pandas as pd
import os 

import methodsHeston, methods


S = 100
K = 80
kappa = 0.01
r = 0.05
rho = 0.3
sigma = 0.1
T = 1
theta = 0.03
v0 = 0.03

print(methodsHeston.priceHeston(S, K, T, r, kappa, theta, rho, sigma, v0, "C"))

n = 10**6

S_table, V_table = methods.Heston_front(n, S, v0, r, rho, kappa, theta, sigma, 1)

print(np.mean(S_table[:, -1]) - 80)
