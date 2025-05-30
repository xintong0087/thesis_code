import numpy as np
from numpy.random import default_rng


def dynamicLapseMultiplier(fundValue, guaranteeValue,
                           U=1, L=0.5, M=1.25, D=1.1):

    """
    This function calculates the dynamic lapse multiplier.

    :param fundValue (np.ndarray): the fund value.
        shape: (M, ), the number of outer paths.
        dtype: float
    :param guaranteeValue (np.ndarray): the guaranteed value.
        shape: (M, ), the number of outer paths.
        dtype: float
    :param U, L, M, D (float): the parameters for dynamic lapse.
    :return (np.ndarray): the dynamic lapse multiplier.
        shape: (M, ), the number of outer paths.
        dtype: float
    """

    eps = 1e-9

    return np.minimum(U, np.maximum(L, 1 - M * (guaranteeValue / (fundValue + eps) - D)))


def RS_transition(state_before: np.ndarray, 
                  transition_prob: np.ndarray, 
                  seed: int) -> np.ndarray:

    """
    This function generates the next state of the regime-switching model.

    :param state_before (np.ndarray): the current state of the model, state 0 or 1.
        shape: (M, ), the number of outer paths.
        dtype: int
    :param transition_prob (np.ndarray): the transition probability vector,
        the first element is the probability of transitioning to regime 0, 
        the second element is the probability of transitioning to regime 1.
        shape: (2, ), the number of regimes.
        dtype: float
    :param seed (int): the seed for the random number generator.
    :return (np.ndarray): the next state of the model, state 0 or 1.
        shape: (M, ), the number of outer paths.
        dtype: int
    """

    rng = default_rng(seed)
    
    num = state_before.shape[0]
    rand = rng.uniform(0, 1, num)

    transition_vector = np.where(state_before == 0, transition_prob[0], transition_prob[1])
    transition_flag = (rand <= transition_vector)
    
    state_after = (state_before + transition_flag) % 2

    return state_after


def sim_outer(M, S_0, drift, volatility, G_0, state_0, transition_prob, maturity,
             withdrawal_rate, eta_g,
             lapse_table, asset_model, contract_type,
             seed_states, seed_diffusion):

    """
    This function generates the outer paths of the asset value, fund value, and guaranteed value.

    :param M (int): the number of outer paths.
    :param S_0 (float): the initial asset value.
    :param drift (float/np.ndarray, GBM/GBM_RS): the drift of the asset.
    :param volatility (float/np.ndarray, GBM/GBM_RS): the volatility of the asset.
    :param G_0 (float): the initial guaranteed value.
    :param state_0 (int): the initial state of the model.
    :param transition_prob (np.ndarray): the transition probability vector,
        the first element is the probability of transitioning to regime 0, 
        the second element is the probability of transitioning to regime 1.
        shape: (2, ), the number of regimes.
        dtype: float
    :param maturity (int): the maturity of the model.
    :param withdrawal_rate (float, GMWB only): the withdrawal rate.
    :param eta_g (float, GMWB only): 
    :param lapse_table (np.ndarray): the lapse table.
        shape: (M, maturity),
        dtype: float
    :param asset_model (str): the model of the asset, either "GBM" or "GBM_RS".
    :param contract_type (str): the type of the contract, either "GMWB" or "GMMB".
    :param seed_states (int):the seed for the random number generator for the states.
    :param seed_diffusion (int): the seed for the random number generator for the diffusion.
    :return states, S_table, F_table, G_table, I_table, survival_rates:
        states: the regime of the outer paths.
            shape: (M, maturity + 1)
            dtype: int
        S_table: the asset value of the outer paths.
            shape: (M, maturity + 1)
            dtype: float
        F_table: the fund value of the outer paths.
            shape: (M, maturity + 1)
            dtype: float
        G_table: the guaranteed value of the outer paths.
            shape: (M, maturity + 1)
            dtype: float
        I_table: the withdrawal amount of the outer paths.
            shape: (M, maturity + 1)
            dtype: float
        survival_rates: the survival rates of the outer paths.
            shape: (M, maturity)
            dtype: float
    """

    # generate the random regimes for the outer simulation
    if asset_model == "GBM_RS":
        states = np.zeros([M, maturity + 1], dtype=int)
        states[:, 0] = state_0
        for n in range(maturity):
            states[:, n + 1] = RS_transition(states[:, n], transition_prob, seed_states)

        # set drift and volatility according to regime, 
        # the last period (maturity) requires no drift and volatility
        drift = np.where(states[:, :maturity] == 0, drift[0], drift[1])
        volatility = np.where(states[:, :maturity] == 0, volatility[0], volatility[1])
    elif asset_model != "GBM":
        raise ValueError("Invalid asset model")

    # generate outer path of the asset
    rng = default_rng(seed_diffusion)
    Z = rng.normal(0, 1, [M, maturity])
    S_table = np.zeros([M, maturity + 1])
    S_table[:, 0] = S_0
    for n in range(maturity):
        S_table[:, n + 1] = S_table[:, n] * np.exp(drift[:, n] + volatility[:, n] * Z[:, n])

    if contract_type == "GMMB":
        # generate survival rates from the lapse table
        survival_rates = np.cumprod(1 - lapse_table, axis=1)

        # generate fund value and guaranteed value
        F_table = S_table * survival_rates
        G_table = G_0 * survival_rates
        I_table = np.zeros_like(F_table)

    elif contract_type == "GMWB":
        
        # initialize the fund value, guaranteed value, and withdrawal amount
        F_table = np.zeros_like(S_table)
        G_table = np.full_like(S_table, G_0)
        I_table = np.zeros_like(S_table)

        # calculate the fund value, guaranteed value, and withdrawal amount
        for t in range(maturity):

            # get the current values
            S_t = S_table[:, t]
            F_t = F_table[:, t]
            G_t = G_table[:, t]
            I_t = I_table[:, t]
            lapse_t = lapse_table[:, t]

            # calculate the dynamic lapse multiplier
            dlapse_multiplier = dynamicLapseMultiplier(F_t, G_t)
            lapse_table[:, t] = lapse_t * dlapse_multiplier

            # calculate the next values
            F_table[:, t + 1] = np.maximum(F_t - I_t, 0) * (S_table[:, t + 1] / S_t) * (1 - eta_g) * (1 - lapse_t * dlapse_multiplier) 
            G_table[:, t + 1] = np.maximum(F_table[:, t + 1], G_t * (1 - lapse_t * dlapse_multiplier))
            I_table[:, t + 1] = G_table[:, t + 1] * withdrawal_rate
        
        # calculate the survival rates
        survival_rates = np.cumprod(1 - lapse_table, axis=1)
                
    else:
        raise ValueError("Invalid contract type")

    return states, S_table, F_table, G_table, I_table, survival_rates
    
    


    
    