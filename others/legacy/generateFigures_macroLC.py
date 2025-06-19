import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
from cycler import cycler

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.locator_params(axis='y', nbins=5)
plt.tight_layout()
plt.subplots_adjust(left=0.5, bottom=0.5)
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

VA_type = "GMWB"
lapse_type = "nolapse"

save_path = f"./trainedModels/{VA_type}_PY/{lapse_type}/"

model_names = ["LSTM_LoCap_LowNoise",
               "LSTM_LoCap_mediumNoise",
               "LSTM_LoCap_highNoise"]

legend_names = ["Proxy - Low Noise", "Proxy - Medium Noise", "Proxy - High Noise"]

df_MSE = pd.DataFrame(index=["Training Error", "Test Error", "True Error", "Training Error SE", "Test Error SE", "True Error SE"],
                      columns=legend_names)

fig_ptm = plt.figure(figsize=(8, 6))
ax_ptm = fig_ptm.add_subplot(1, 1, 1)

fig_CVaR = plt.figure(figsize=(8, 6))
ax_CVaR = fig_CVaR.add_subplot(1, 1, 1)

start = 2500
end = int(start * 4)
step = (end - start) // 30
margin = (np.arange(start, end, step) / start - 1) * 0.05

for mdl, lgd in zip(model_names, legend_names):

   MSE_train = np.load(save_path + f"{mdl}_MSE_train.npy")
   MSE_test = np.load(save_path + f"{mdl}_MSE_test.npy")
   MSE = np.load(save_path + f"{mdl}_MSE.npy")
   ptm = pd.read_csv(save_path + f"{mdl}_percent.csv").iloc[:, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
      17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
   CVaR = pd.read_csv(save_path + f"{mdl}_CVaR.csv").iloc[:, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
      17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]

   df_MSE.loc["Training Error", lgd] = np.mean(MSE_train)
   df_MSE.loc["Test Error", lgd] = np.mean(MSE_test)
   df_MSE.loc["True Error", lgd] = np.mean(MSE)
   df_MSE.loc["Training Error SE", lgd] = np.std(MSE_train)
   df_MSE.loc["Test Error SE", lgd] = np.std(MSE_test)
   df_MSE.loc["True Error SE", lgd] = np.std(MSE)

   ptm_mean = ptm.mean(axis=1)
   ptm_SE = ptm.std(axis=1)
   ptm_upper = ptm.quantile(0.96, axis=1)
   ptm_lower = ptm.quantile(0.04,axis=1)

   ax_ptm.plot(margin, ptm_mean, label=lgd)
   ax_ptm.fill_between(margin, ptm_upper, ptm_lower, alpha=0.2)

   CVaR_mean = CVaR.mean(axis=1)
   CVaR_SE = CVaR.std(axis=1)
   CVaR_upper = CVaR.quantile(0.96, axis=1)
   CVaR_lower = CVaR.quantile(0.04, axis=1)

   ax_CVaR.plot(margin, CVaR_mean, label=lgd)
   ax_CVaR.fill_between(margin, CVaR_upper, CVaR_lower, alpha=0.2)

ax_ptm.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax_ptm.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax_ptm.set_ylim([0.35, 1])
ax_ptm.set_xticks(np.arange(0, 0.16, 0.05))
ax_ptm.set_xlabel("Safety Margin")
ax_ptm.set_ylabel("Percent of Matches")
ax_ptm.legend(loc="lower right")
ax_ptm.set_title("LSTM LoCap")

ax_CVaR.set_xticks(np.arange(0, 0.16, 0.05))
ax_CVaR.set_xlabel("Safety Margin")
ax_CVaR.set_ylabel("CVaR")
ax_CVaR.legend(loc="lower right")
ax_CVaR.set_title("LSTM LoCap")


fig_ptm.savefig(f"./figures/{VA_type}/percent_macroLC.png")
fig_CVaR.savefig(f"./figures/{VA_type}/CVaR_macroLC.png")

print(df_MSE)
df_MSE.to_csv(f"./figures/{VA_type}/MSE_macroLC.csv")

