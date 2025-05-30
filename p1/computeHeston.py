import numpy as np
import pandas as pd
import os 
from tqdm import tqdm

import methodsHeston, methods


def ComputeTrueLossEC(n_front, S0, V0, K, 
                      mu, rho, kappa, theta, sigma, r, tau, T, 
                      level_list=[0.8, 0.9, 0.95, 0.99, 0.996]):

    portfolio_value_0 = 0

    for j in range(len(K)):
        portfolio_value_0 = portfolio_value_0 + methodsHeston.priceHeston(S0, K[j], T-tau, r, kappa, theta, rho, sigma, V0, "C")

    print("European Call: Simulating Front Paths...")
    S, V = methods.Heston_front(n_front, S0, V0, mu, rho, kappa, theta, sigma, r, tau)

    print("Calculating Value...")
    call_tau = np.zeros(n_front)
    for j in range(len(K)):
        for k in tqdm(range(n_front)):
            call_tau[k] += methodsHeston.priceHeston(S[k, -1], K[j], T-tau, r, kappa, theta, rho, sigma, V[k, -1], "C")
    print("End of European Call.")

    loss_true = portfolio_value_0 - call_tau

    loss_true.sort()

    L0 = loss_true[int(np.ceil(0.9 * n_front)) - 1]

    indicator_true = 0.1
    hockey_true = np.mean(np.maximum(loss_true - L0, 0))
    quadratic_true = np.mean((loss_true - L0) ** 2)

    VaR = {}
    CVaR = {}
    for level in level_list:
        VaR[level] = loss_true[int(np.ceil(level * n_front)) - 1] 
        CVaR[level] = np.mean(loss_true[loss_true >= VaR[level]])

    return indicator_true, hockey_true, quadratic_true, VaR, CVaR


n = 10**7
S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1
sigma = 0.1

rho = 0.3
kappa = 2
theta = 0.04
V0 = 0.04

indicator, hockeyStick, quadratic, VaR, CVaR = ComputeTrueLossEC(n, S_0, V0, K, 
                      mu, rho, kappa, theta, sigma, r, tau, T)

result_true = [indicator, hockeyStick, quadratic]
label_true = ["indicator", "hockeyStick", "quadratic"]
for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
    result_true.append(VaR[level])
    result_true.append(CVaR[level])
    label_true.append("VaR_" + str(level))
    label_true.append("CVaR_" + str(level))

df = pd.DataFrame(result_true,
                    index=label_true,
                    columns=["True Value"]).T

saveFolder = "./trueValues/"
if os.path.exists(saveFolder) == False:
    os.makedirs(saveFolder)

df.to_csv(f"{saveFolder}trueValue_European_1.csv")