import numpy as np
import pandas as pd
import os


def cvStudy(d, S_0, K, mu, sigma, r, tau, T, h=3/50, H=120, L0=100, 
            optionName="European", model="Kernel", level_list=[0.8, 0.9, 0.95, 0.99, 0.996],
            n_rep=1000):
    
    if model == "Kernel":

        M = 10000
        N = 10

        para_list = []
        result_table = {}

        if optionName == "European":
            from bsKernel import kernelEC 
            from bsKernel import cvEC 

            for n in range(n_rep):

                k_opt = cvEC(M, N, d, S_0, K, mu, sigma, r, tau, T)
                para_list.append(k_opt)

                loss = kernelEC(M, N, d, S_0, K, mu, sigma, r, tau, T, k_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                if (n + 1) % 100 == 0:
                    print(f"{n+1} iterations finished!")

        elif optionName == "Asian":
            from bsKernel import kernelGA 
            from bsKernel import cvGA

            for n in range(n_rep):

                k_opt = cvGA(M, N, d, S_0, K, mu, sigma, r, tau, T, h)
                para_list.append(k_opt)

                loss = kernelGA(M, N, d, S_0, K, mu, sigma, r, tau, T, h, k_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM
                
                if (n + 1) % 100 == 0:
                    print(f"{n+1} iterations finished!")

        elif optionName == "BarrierUp":
            from bsKernel import kernelBU 
            from bsKernel import cvBU

            for n in range(n_rep):

                k_opt = cvBU(M, N, d, S_0, K, mu, sigma, r, tau, T, H, h)
                para_list.append(k_opt)

                loss = kernelBU(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H, k_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                if (n + 1) % 100 == 0:
                    print(f"{n+1} iterations finished!")

        elif optionName == "BarrierDown":
            from bsKernel import kernelBD 
            from bsKernel import cvBD

            for n in range(n_rep):
                    
                k_opt = cvBD(M, N, d, S_0, K, mu, sigma, r, tau, T, H, h)
                para_list.append(k_opt)

                loss = kernelBD(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H, k_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                if (n + 1) % 100 == 0:
                    print(f"{n+1} iterations finished!")

    elif model == "KRR":

        M = 10000
        N = 10

        para_list = {}
        result_table = {}

        if optionName == "European":
            from bsKRR import krrEC 
            from bsKRR import cvEC 

            for n in range(n_rep):

                alpha_opt, l_opt, nu_opt = cvEC(M, N, d, S_0, K, mu, sigma, r, tau, T)
                
                para_list[n] = [alpha_opt, l_opt, nu_opt]

                loss = krrEC(M, N, d, S_0, K, mu, sigma, r, tau, T, alpha_opt, l_opt, nu_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                print(f"{n+1} iterations finished!")

        elif optionName == "Asian":
            from bsKRR import krrGA 
            from bsKRR import cvGA

            for n in range(n_rep):
                    
                alpha_opt, l_opt, nu_opt = cvGA(M, N, d, S_0, K, mu, sigma, r, tau, T, h)
                para_list[n] = [alpha_opt, l_opt, nu_opt]

                loss = krrGA(M, N, d, S_0, K, mu, sigma, r, tau, T, h, alpha_opt, l_opt, nu_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                print(f"{n+1} iterations finished!")

        elif optionName == "BarrierUp":
            from bsKRR import krrBU 
            from bsKRR import cvBU

            for n in range(n_rep):
                        
                alpha_opt, l_opt, nu_opt = cvBU(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H)
                para_list[n] = [alpha_opt, l_opt, nu_opt]

                loss = krrBU(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H, alpha_opt, l_opt, nu_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                if (n + 1) % 5 == 0:
                    print(f"{n+1} iterations finished!")

        elif optionName == "BarrierDown":
            from bsKRR import krrBD 
            from bsKRR import cvBD

            for n in range(n_rep):

                alpha_opt, l_opt, nu_opt = cvBD(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H)
                para_list[n] = [alpha_opt, l_opt, nu_opt]
                
                loss = krrBD(M, N, d, S_0, K, mu, sigma, r, tau, T, h, H, alpha_opt, l_opt, nu_opt)

                loss.sort()

                indicator = np.mean((loss > L0))
                hockey = np.mean(np.maximum(loss - L0, 0))
                quadratic = np.mean((loss - L0) ** 2)

                VaR = {}
                CVaR = {}
                for level in level_list:
                    VaR[level] = loss[int(np.ceil(level * M)) - 1] 
                    CVaR[level] = np.mean(loss[loss >= VaR[level]])
                
                RM = [indicator, hockey, quadratic]
                for level in [0.8, 0.9, 0.95, 0.99, 0.996]:
                    RM.append(VaR[level])
                    RM.append(CVaR[level])

                result_table[n] = RM

                print(f"{n+1} iterations finished!")

    else:
        print("Unable to load module!")
        exit()

    return result_table, para_list


model = input("Please choose a model from Kernel and KRR: ")
optionName = input("Option Type? Please enter one of: European, Asian, BarrierUp, BarrierDown:")
d = int(input("Please enter dimension:"))

trueValueFolder = "./trueValues/"
saveFolder = "./result/"

if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

S_0 = 100
K = [90, 100, 110]
mu = 0.08
r = 0.05
tau = 3/50
T = 1

n_rep = 1000

if (d<1) or (d>100):
    print("Invalid Dimension!")
    exit()

if optionName == "European":
    sigma = 0.1
    H = None
    h = None
    
elif optionName == "Asian":
    sigma = 0.3
    H = None
    h = T / 50
    tau = 3 * h

elif optionName == "BarrierUp":
    sigma = 0.2
    H = 120
    h = T / 50
    tau = 3 * h

elif optionName == "BarrierDown":
    sigma = 0.2
    H = 90
    h = T / 50
    tau = 3 * h

else:
    print("Invalid Option Type!")
    exit()


trueValues = pd.read_csv(f"{trueValueFolder}trueValue_{optionName}_{d}.csv")

RM_names = trueValues.columns.values[1:]

L0 = trueValues["VaR_0.9"].values[0]
trueValues = np.array(trueValues).flatten()[1:]

res_table, para_table = cvStudy(d, S_0, K, mu, sigma, r, tau, T, h, H, L0, optionName, model, n_rep=n_rep)

res_table = pd.DataFrame(res_table, index=RM_names).T
if model == "Kernel":
    para_table = pd.DataFrame(para_table, columns=["Optimal k"])
elif model == "KRR":
    para_table = pd.DataFrame(para_table, index=["Optimal alpha", "Optimal l", "Optimal nu"]).T

para_table.to_csv(f"{saveFolder}CVpara_{optionName}_{d}_{model}.csv")
res_table.to_csv(f"{saveFolder}CVresult_{optionName}_{d}_{model}.csv")