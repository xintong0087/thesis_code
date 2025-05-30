import numpy as np
from scipy.stats.distributions import norm


def generate_cor_mat(d, rho):
    return np.ones((d, d)) * rho + np.identity(d) * (1 - rho)


def Heston_front(n_front, S_0, V_0, mu, rho, kappa, theta, sigma, tau, step_size=1/253):

    n_step = int(tau // step_size) + 1
    mu_Z = np.array([0, 0])
    cov_Z = np.array([[1, rho], [rho, 1]])
    
    S_front = np.zeros([n_front, n_step + 1])
    S_front[:, 0] = np.array(S_0)
    V = np.zeros([n_front, n_step + 1])
    V[:, 0] = np.array(V_0)

    Z = np.random.multivariate_normal(mu_Z, cov_Z, [n_front, n_step])

    for k in range(1, n_step + 1):
        S_front[:, k] = S_front[:, k - 1] * np.exp((mu - (1/2) * V[:, k-1]) * step_size 
                                                   + np.sqrt(V[:, k-1] * step_size) 
                                                   * Z[:, k-1, 0])
        V[:, k] = np.maximum(V[:, k - 1] 
                             + kappa * (theta - V[:, k - 1]) * step_size 
                             + sigma * np.sqrt(V[:, k - 1] * step_size) * Z[:, k-1, 1], 0)

    return S_front, V


def Heston_back(n_front, n_back, S_tau, V_tau, r, rho, kappa, theta, sigma, T, step_size=1/253):

    n_step = int(T // step_size) + 1
    mu_Z = np.array([0, 0])
    cov_Z = np.array([[1, rho], [rho, 1]])

    S_back = np.zeros([n_front, n_back, n_step + 1])
    V = np.zeros([n_front, n_back, n_step + 1])
    for j in range(n_back):
        S_back[:, j, 0] = S_tau[:, -1]
        V[:, j, 0] = V_tau[:, -1]
    
    Z = np.random.multivariate_normal(mu_Z, cov_Z, [n_front, n_back, n_step])

    for k in range(1, n_step + 1):

        S_back[:, :, k] = S_back[:, :, k - 1] * np.exp((r - (1/2) * V[:, :, k-1]) * step_size 
                                                        + np.sqrt(V[:, :, k-1] * step_size) 
                                                        * Z[:, :, k-1, 0])
        V[:, :, k] = np.maximum(V[:, :, k - 1] 
                                + kappa * (theta - V[:, :, k - 1] ) * step_size 
                                + sigma * np.sqrt(V[:, :, k - 1]  * step_size) * Z[:, :, k-1, 1], 0)
    
    return S_back, V


def GBM_front(n_front, d, S_0, drift_vec, diffusion_mat, tau, step_size=1 / 253, path=False):
    drift_vec = np.full(d, drift_vec)
    diffusion_mat = np.array(diffusion_mat)
    A = np.linalg.cholesky(diffusion_mat)

    if path:

        n_step = int(tau // step_size) + 1
        S_front = np.zeros([d, n_front, n_step + 1])
        S_front[:, :, 0] = np.array(S_0)

        for k in range(1, n_step + 1):
            Z = np.random.normal(0, 1, [d, n_front])
            drift = (drift_vec - (1 / 2) * np.diagonal(diffusion_mat)) * step_size
            drift = drift.reshape(-1, 1)
            diffusion = np.sqrt(step_size) * np.matmul(A, Z)
            S_front[:, :, k] = S_front[:, :, k - 1] * np.exp(drift + diffusion)

    else:

        S_front = np.zeros([d, n_front])

        for i in range(n_front):
            Z = np.random.normal(0., 1., d)
            drift = (drift_vec - 0.5 * np.diagonal(diffusion_mat)) * tau
            diffusion = np.matmul(A, Z) * np.sqrt(tau)
            S_front[:, i] = S_0 * np.exp(drift + diffusion)

    return S_front


def GBM_back(n_front, n_back, d, S_tau, drift, diffusion, T, step_size=1 / 253, path=False):
    drift_vec = np.full(d, drift)
    diffusion_mat = np.array(diffusion)
    A = np.linalg.cholesky(diffusion_mat)

    if path:
        n_step = int(T // step_size) + 1
        S_back = np.zeros([d, n_front, n_back, n_step + 1])
        for j in range(n_back):
            S_back[:, :, j, 0] = S_tau[:, :, -1]

        for i in range(n_front):
            for k in range(1, n_step + 1):
                Z = np.random.normal(0, 1, [d, n_back])
                drift = (drift_vec - (1 / 2) * np.diagonal(diffusion_mat)) * step_size
                drift = drift.reshape(-1, 1)
                diffusion = np.sqrt(step_size) * np.matmul(A, Z)
                S_back[:, i, :, k] = S_back[:, i, :, k - 1] * np.exp(drift + diffusion)

    else:
        S_back = np.zeros([d, n_front, n_back])

        for i in range(n_front):
            for j in range(n_back):
                Z = np.random.normal(0., 1., d)
                drift = (drift_vec - 0.5 * np.diagonal(diffusion_mat)) * T
                diffusion = np.matmul(A, Z) * np.sqrt(T)
                S_back[:, i, j] = S_tau[:, i] * np.exp(drift + diffusion)

    return S_back


def price_CP(S, T, sigma, r, K, q, option_type, position):
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S / K) + (r + sigma ** 2 / 2) * T)
    d2 = d1 - (sigma * np.sqrt(T))

    if option_type == 'C':
        price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    else:
        price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S

    if position == "short":
        price = - price

    return price


def price_Asian_G(S, T, sigma, r, K, option_type="C", position="long",
                  continuous=False, **kwargs):
    n_interval = kwargs["args"]
    sigma_G = sigma / n_interval * np.sqrt((n_interval + 1) * (2 * n_interval + 1) / 6)
    delta = 1 / 2 * (
                (r * (n_interval - 1) / n_interval) + sigma ** 2 * (n_interval + 1) / (2 * n_interval) - sigma ** 2 * (
                    n_interval + 1) * (2 * n_interval + 1) / (6 * n_interval ** 2))

    d1 = (np.log(S / K) + (r - delta + sigma_G ** 2 / 2) * T) / (sigma_G * np.sqrt(T))
    d2 = d1 - sigma_G * np.sqrt(T)
    if option_type == "C":
        price = np.exp(-delta * T) * (S * norm.cdf(d1)) - K * norm.cdf(d2) * np.exp(-r * T)
    else:
        price = np.exp(-delta * T) * (-S * norm.cdf(-d1)) + K * norm.cdf(-d2) * np.exp(-r * T)

    if position == "short":
        price = - price

    return price


def price_barrier(S, K, T, sigma, r, H, q, R,
                  option_type, position="long"):

    S = np.array(S)
    length_S = S.shape[0]
    phi = np.zeros(length_S)
    nu = np.zeros(length_S)

    phi[(option_type[:, 1] == "U") & (option_type[:, 2] == "C")] = 1
    nu[(option_type[:, 1] == "U") & (option_type[:, 2] == "C")] = -1

    phi[(option_type[:, 1] == "U") & (option_type[:, 2] == "P")] = -1
    nu[(option_type[:, 1] == "U") & (option_type[:, 2] == "P")] = -1

    phi[(option_type[:, 1] == "D") & (option_type[:, 2] == "C")] = 1
    nu[(option_type[:, 1] == "D") & (option_type[:, 2] == "C")] = 1

    phi[(option_type[:, 1] == "D") & (option_type[:, 2] == "P")] = -1
    nu[(option_type[:, 1] == "D") & (option_type[:, 2] == "P")] = 1

    sigma_sq = sigma ** 2
    sigma_T = sigma * np.sqrt(T)

    mu = (r - q - sigma_sq / 2) / sigma_sq
    _lambda = np.sqrt(mu ** 2 + 2 * r / sigma_sq)
    z = np.log(H / S) / sigma_T + _lambda * sigma_T

    mu_sigma_T = (1 + mu) * sigma_T

    x1 = np.log(S / K) / sigma_T + mu_sigma_T
    x2 = np.log(S / H) / sigma_T + mu_sigma_T
    y1 = np.log(H ** 2 / (S * K)) / sigma_T + mu_sigma_T
    y2 = np.log(H / S) / sigma_T + mu_sigma_T

    A = phi * S * np.exp(-q * T) * norm.cdf(phi * x1, 0, 1) \
        - phi * K * np.exp(-r * T) * norm.cdf(phi * (x1 - sigma_T), 0, 1)
    B = phi * S * np.exp(-q * T) * norm.cdf(phi * x2, 0, 1) \
        - phi * K * np.exp(-r * T) * norm.cdf(phi * (x2 - sigma_T), 0, 1)
    C = phi * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(nu * y1, 0, 1) \
        - phi * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(nu * (y1 - sigma_T), 0, 1)
    D = phi * S * np.exp(-q * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(nu * y2, 0, 1) \
        - phi * K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(nu * (y2 - sigma_T), 0, 1)
    E = R * np.exp(-r * T) * (norm.cdf(nu * (x2 - sigma_T))
                              - (H / S) ** (2 * mu) * norm.cdf(nu * (y2 - sigma_T)))
    F = R * ((H / S) ** (mu + _lambda) * (norm.cdf(nu * z))
             - (H / S) ** (mu - _lambda) * norm.cdf(nu * (z - 2 * _lambda * sigma_T)))

    price = np.zeros(length_S)

    # Up and In Options
    IUC_aboveBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[IUC_aboveBarrier_flag] = A[IUC_aboveBarrier_flag] + E[IUC_aboveBarrier_flag]

    IUC_belowBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[IUC_belowBarrier_flag] = B[IUC_belowBarrier_flag] - C[IUC_belowBarrier_flag] + D[IUC_belowBarrier_flag] \
                                   + E[IUC_belowBarrier_flag]

    IUP_aboveBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[IUP_aboveBarrier_flag] = A[IUP_aboveBarrier_flag] - B[IUP_aboveBarrier_flag] + D[IUP_aboveBarrier_flag] \
                                   + E[IUP_aboveBarrier_flag]

    IUP_belowBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[IUP_belowBarrier_flag] = C[IUP_belowBarrier_flag] + E[IUP_belowBarrier_flag]

    # Down and In Options
    IDC_aboveBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[IDC_aboveBarrier_flag] = C[IDC_aboveBarrier_flag] + E[IDC_aboveBarrier_flag]

    IDC_belowBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[IDC_belowBarrier_flag] = A[IDC_belowBarrier_flag] - B[IDC_belowBarrier_flag] + D[IDC_belowBarrier_flag] \
                                   + E[IDC_belowBarrier_flag]

    IDP_aboveBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[IDP_aboveBarrier_flag] = B[IDP_aboveBarrier_flag] - C[IDP_aboveBarrier_flag] + D[IDP_aboveBarrier_flag] \
                                   + E[IDP_aboveBarrier_flag]

    IDP_belowBarrier_flag = (option_type[:, 0] == "I") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[IDP_belowBarrier_flag] = A[IDP_belowBarrier_flag] + E[IDP_belowBarrier_flag]

    # Up and Out Options
    OUC_aboveBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[OUC_aboveBarrier_flag] = F[OUC_aboveBarrier_flag]

    OUC_belowBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[OUC_belowBarrier_flag] = A[OUC_belowBarrier_flag] - B[OUC_belowBarrier_flag] + C[OUC_belowBarrier_flag] \
                                   - D[OUC_belowBarrier_flag] + F[OUC_belowBarrier_flag]

    OUP_aboveBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[OUP_aboveBarrier_flag] = B[OUP_aboveBarrier_flag] - D[OUP_aboveBarrier_flag] + F[OUP_aboveBarrier_flag]

    OUP_belowBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "U") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[OUP_belowBarrier_flag] = A[OUP_belowBarrier_flag] - C[OUP_belowBarrier_flag] + F[OUP_belowBarrier_flag]

    # Down and Out Options
    ODC_aboveBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S >= H)
    price[ODC_aboveBarrier_flag] = A[ODC_aboveBarrier_flag] - C[ODC_aboveBarrier_flag] + F[ODC_aboveBarrier_flag]

    ODC_belowBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "C") \
                            & (S < H)
    price[ODC_belowBarrier_flag] = B[ODC_belowBarrier_flag] - D[ODC_belowBarrier_flag] + F[ODC_belowBarrier_flag]

    ODP_aboveBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S >= H)
    price[ODP_aboveBarrier_flag] = A[ODP_aboveBarrier_flag] - B[ODP_aboveBarrier_flag] + C[ODP_aboveBarrier_flag] \
                                   - D[ODP_aboveBarrier_flag] + F[ODP_aboveBarrier_flag]

    ODP_belowBarrier_flag = (option_type[:, 0] == "O") \
                            & (option_type[:, 1] == "D") \
                            & (option_type[:, 2] == "P") \
                            & (S < H)
    price[ODP_belowBarrier_flag] = F[ODP_belowBarrier_flag]

    if position == "short":
        price = - price

    return price


def price_barrier_up(n_front, max_outer, outer_sample, K, t, sigma, r, U):
    price = (max_outer < U) * price_barrier(outer_sample, K, t, sigma, r, U, 0, 0,
                                            np.full([n_front, 3], ["O", "U", "C"]))

    return price


def price_barrier_down(n_front, min_outer, outer_sample, K, t, sigma, r, D):
    price = (min_outer > D) * price_barrier(outer_sample, K, t, sigma, r, D, 0, 0,
                                            np.full([n_front, 3], ["O", "D", "C"]))

    return price


def price_Asian_G_tau(S_tau, T, sigma, r, K, continuous=True, **kwargs):
    n_interval_outer = kwargs["args"][0]
    n_interval_inner = kwargs["args"][1]
    n_interval = n_interval_outer + n_interval_inner

    delta_G = r - (r - (sigma ** 2 / 2)) * (1 / n_interval) ** 2 * (n_interval_inner * (n_interval_inner + 1)) / 2 \
              - (1 / 2) * (n_interval_inner / n_interval) ** 2 * sigma ** 2 * (1 / n_interval) * (
                          ((n_interval_inner + 1) * (2 * n_interval_inner + 1)) / (6 * n_interval_inner))
    sigma_G = (n_interval_inner / n_interval) * sigma * np.sqrt(1 / n_interval) * np.sqrt(
        ((n_interval_inner + 1) * (2 * n_interval_inner + 1)) / (6 * n_interval_inner))
    d1 = (np.log(S_tau[:, -1] ** (n_interval_inner / n_interval) * np.prod(S_tau, axis=1) ** (1 / n_interval) / K)
          + (r - delta_G + sigma_G ** 2 / 2)) / sigma_G
    d2 = d1 - sigma_G

    price = np.exp(n_interval_outer / n_interval * r) * np.prod(S_tau, axis=1) ** (1 / n_interval) * (
                np.exp(-delta_G) * S_tau[:, -1] ** (n_interval_inner / n_interval) * norm.cdf(d1)
                - K * np.exp(-r) / np.prod(S_tau, axis=1) ** (1 / n_interval) * norm.cdf(d2))

    return price


def Laguerre_polynomial(X, degree=3):

    X_norm = (X - X.mean(axis=1).reshape(-1, 1)) / X.std(axis=1).reshape(-1, 1)
    # Normalization

    L_0 = np.exp(-X_norm / 2)
    L_1 = np.exp(-X_norm / 2) * (1 - X_norm)
    L_2 = np.exp(-X_norm / 2) * (1 - 2 * X_norm + (1 / 2) * (X_norm ** 2))
    L_3 = np.exp(-X_norm / 2) * (1 - 3 * X_norm + (3 / 2) * (X_norm ** 2) - (1 / 6) * (X_norm ** 3))

    L = [L_1, L_2, L_3]

    X_train = L_0
    for k in range(degree):
        X_train = np.concatenate([X_train, L[k]], axis=0)
        # X_train.shape = (d*5, n_front)

    return X_train


def generate_basis(sample_outer, option_type="Vanilla",
                   basis_function="Laguerre",
                   sample_max=1, sample_min=0):

    if option_type == "Asian":

        S_tau = sample_outer[:, :, -1]
        # S_tau
        # sample_outer.shape = [d, n_front, n_step]

        geometric_sum_outer = np.prod(sample_outer[:, :, 1:], axis=2)

        X = np.concatenate([S_tau, geometric_sum_outer], axis=0)

        if basis_function == "Laguerre":
            X_train = Laguerre_polynomial(X)
        else:
            X_train = X

        X_train = X_train.T

    elif option_type == "Barrier_U":

        S_tau = sample_outer[:, :, -1]

        X_U = np.concatenate([S_tau, sample_max], axis=0)

        if basis_function == "Laguerre":
            X_train_U = Laguerre_polynomial(X_U)
        else:
            X_train_U = Laguerre_polynomial(X_U)

        X_train = X_train_U.T

    elif option_type == "Barrier_D":

        S_tau = sample_outer[:, :, -1]

        X_D = np.concatenate([S_tau, sample_min], axis=0)

        if basis_function == "Laguerre":
            X_train_D = Laguerre_polynomial(X_D)
        else:
            X_train_D = Laguerre_polynomial(X_D)

        X_train = X_train_D.T

    else:

        X = sample_outer

        if basis_function == "Laguerre":
            X_train = Laguerre_polynomial(X)
        else:
            X_train = X

        X_train = X_train.T

    return X_train


def compute_Weight_D(S_0, mu, r, sigma, h, tau, sample_inner_tau):
    return (np.log(sample_inner_tau / S_0) - (mu - (1 / 2) * sigma ** 2) * tau - (
                r - (1 / 2) * sigma ** 2) * h) ** 2 / (2 * sigma ** 2 * (tau + h))


def compute_Weight_U(sample_inner_tau, sample_outer_test, r, sigma, h):

    log_ratio = np.log(sample_inner_tau.reshape(-1, 1) / sample_outer_test.reshape(1, -1))

    return (log_ratio - (r - sigma ** 2 / 2) * h) ** 2 / (2 * sigma ** 2 * h)

