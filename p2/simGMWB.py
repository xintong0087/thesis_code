import numpy as np
from numpy.random import default_rng

import lapse


def RS_transition(state_before, transition_prob, seed):

    rng = np.random.default_rng(seed)
    
    rand = rng.uniform(size=state_before.shape[0])
    transition_vector = np.where(state_before == 0, transition_prob[0], transition_prob[1])
    transition_flag = rand <= transition_vector

    state_after = (state_before + transition_flag) % 2

    return state_after


def RS_outer(M, S_0, drift, volatility, GV_0, initial_state, transition_prob, n_steps,
             withdrawal_rate, eta_g,
             lapse_table, lapse_flag, dynamic_flag,
             seed_states, seed_diffusion, seed_dlapse):
    
    # Generate the RS states for outer simulation
    states = np.zeros([M, n_steps + 1], dtype=int)
    states[:, 0] = initial_state
    rng_states = np.random.default_rng(seed_states)
    for n in range(n_steps):
        states[:, n + 1] = RS_transition(states[:, n], transition_prob, rng_states)

    # Set drift and diffusion according to regime, the last period (maturity) requires no drift and volatility
    drift_vector = np.where(states[:, :n_steps] == 0, drift[0], drift[1])
    volatility_vector = np.where(states[:, :n_steps] == 0, volatility[0], volatility[1])

    # Generate outer path of the asset
    rng = np.random.default_rng(seed_diffusion)
    Z = rng.standard_normal([M, n_steps])
    S_table = S_0 * np.cumprod(np.exp(drift_vector + volatility_vector * Z), axis=1)
    S_table = np.insert(S_table, 0, S_0, axis=1)

    FV_table = np.copy(S_table)
    GV_table = np.full_like(FV_table, GV_0)
    I_table = np.zeros_like(GV_table)
    
    if lapse_flag:
        if dynamic_flag:
            # Simulating with dynamic lapse
            for j in range(S_table.shape[1] - 1):
                dlapse_multiplier = lapse.cal_dynamicLapseMultiplier(FV_table[:, j], GV_table[:, j])
                lapse_table[:, j] = lapse_table[:, j] * dlapse_multiplier

                FV_table[:, j + 1] = np.maximum(FV_table[:, j] - I_table[:, j], 0) * (S_table[:, j + 1] / S_table[:, j]) \
                    * (1 - eta_g) * (1 - lapse_table[:, j]) 
                GV_table[:, j + 1] = np.maximum(FV_table[:, j + 1], GV_table[:, j] * (1 - lapse_table[:, j]))
                I_table[:, j + 1] = GV_table[:, j + 1] * withdrawal_rate
            
        else:
            # Simulating with static lapse
            for j in range(S_table.shape[1] - 1):
                FV_table[:, j + 1] = np.maximum(FV_table[:, j] - I_table[:, j], 0) * (S_table[:, j + 1] / S_table[:, j]) \
                    * (1 - eta_g) * (1 - lapse_table[:, j])
                GV_table[:, j + 1] = np.maximum(FV_table[:, j + 1], GV_table[:, j] * (1 - lapse_table[:, j]))
                I_table[:, j + 1] = GV_table[:, j + 1] * withdrawal_rate  
                
        survival_rates = np.cumprod(1 - lapse_table, axis=1)        

    else:
        # Simulating with no lapse
        for j in range(S_table.shape[1] - 1):
            FV_table[:, j + 1] = np.maximum(FV_table[:, j] - I_table[:, j], 0) * (S_table[:, j + 1] / S_table[:, j]) * (1 - eta_g) 
            GV_table[:, j + 1] = np.maximum(FV_table[:, j + 1], GV_table[:, j])
            I_table[:, j + 1] = GV_table[:, j + 1] * withdrawal_rate

        survival_rates = np.ones_like(states)

    return states, S_table, FV_table, GV_table, I_table, survival_rates


def RS_inner(N, stockValue, fundValue, guaranteeValue, state, survival_rate_outer,
             r, drift, volatility, n_steps, periodRemain,
             withdrawal_rate, eta_g, eta_n,
             base_lapse, lapse_flag, dynamic_flag,
             transition_prob, seed_states, seed_diffusion, seed_dlapse):

    if (periodRemain == 0) or (fundValue <= withdrawal_rate * guaranteeValue):
        delta = 0
        delta_se = 0
        return delta, delta_se

    # Generate RS states and asset prices
    states = np.zeros([N, periodRemain + 1], dtype=int)
    states[:, 0] = state
    for n in range(periodRemain):
        states[:, n + 1] = RS_transition(states[:, n], transition_prob, seed_states)

    drift_vector = np.where(states[:, :periodRemain] == 0, drift[0], drift[1])
    volatility_vector = np.where(states[:, :periodRemain] == 0, volatility[0], volatility[1])

    rng = np.random.default_rng(seed_diffusion)
    Z = rng.standard_normal([N, periodRemain])
    S_table = stockValue * np.cumprod(np.exp(drift_vector + volatility_vector * Z), axis=1)
    S_table = np.insert(S_table, 0, stockValue, axis=1)

    FV_table = np.copy(S_table)
    GV_table = np.full_like(FV_table, guaranteeValue)
    if periodRemain == n_steps:
        I_table = np.zeros_like(GV_table)
    else:
        I_table = np.copy(GV_table) * withdrawal_rate

    # Calculate survival probabilities and adjust cashflows
    if lapse_flag:
        if dynamic_flag:
            dynamic_lapse_multiplier = lapse.cal_dynamicLapseMultiplier(FV_table[:, :-1], GV_table[:, :-1])
        else:
            dynamic_lapse_multiplier = 1

        inner_baseLapse = base_lapse[-periodRemain:].reshape(1, -1)
        lapse_rate = dynamic_lapse_multiplier * inner_baseLapse
    else:
        lapse_rate = np.zeros_like(states)

    # Calculate deltas and standard errors
    dFV_table = FV_table / S_table
    dGV_table = np.zeros_like(dFV_table)
    dI_table = np.zeros_like(dFV_table)

    if lapse_flag:
        lapse_rate = base_lapse[-periodRemain:].reshape(1, -1)

        if dynamic_flag:
            # Simulating with dynamic lapse
            for n in range(1, periodRemain + 1):

                dlapse_multiplier = lapse.cal_dynamicLapseMultiplier(FV_table[:, n - 1], GV_table[:, n - 1])

                FV_table[:, n] = np.maximum(FV_table[:, n - 1] - I_table[:, n - 1], 0) * (S_table[:, n] / S_table[:, n - 1]) \
                    * (1 - eta_g) * (1 - lapse_rate[:, n - 1]) * dlapse_multiplier
                GV_table[:, n] = np.maximum(FV_table[:, n], GV_table[:, n - 1] * (1 - lapse_rate[:, n - 1]) * dlapse_multiplier)
                I_table[:, n] = GV_table[:, n] * withdrawal_rate

                mask = (I_table[:, n - 1] < FV_table[:, n - 1])
                dFV_table[:, n] = mask * (dFV_table[:, n - 1] - dI_table[:, n - 1]) \
                                * (S_table[:, n] / (S_table[:, n - 1])) * (1 - eta_g)
                flag_guarantee = (GV_table[:, n - 1] >= FV_table[:, n])
                dGV_table[:, n] = (1 - flag_guarantee) * dFV_table[:, n] + flag_guarantee * dGV_table[:, n - 1]
                dI_table[:, n] = withdrawal_rate * dGV_table[:, n]

        else:
            # Simulating with static lapse
            for n in range(1, periodRemain + 1):
                FV_table[:, n] = np.maximum(FV_table[:, n - 1] - I_table[:, n - 1], 0) * (S_table[:, n] / S_table[:, n - 1]) \
                    * (1 - eta_g) * (1 - lapse_rate[:, n - 1])
                GV_table[:, n] = np.maximum(FV_table[:, n], GV_table[:, n - 1] * (1 - lapse_rate[:, n - 1]))
                I_table[:, n] = GV_table[:, n] * withdrawal_rate

                mask = (I_table[:, n - 1] < FV_table[:, n - 1])
                dFV_table[:, n] = mask * (dFV_table[:, n - 1] - dI_table[:, n - 1]) \
                                * (S_table[:, n] / (S_table[:, n - 1])) * (1 - eta_g)
                flag_guarantee = (GV_table[:, n - 1] >= FV_table[:, n])
                dGV_table[:, n] = (1 - flag_guarantee) * dFV_table[:, n] + flag_guarantee * dGV_table[:, n - 1]
                dI_table[:, n] = withdrawal_rate * dGV_table[:, n]
    else:
        # Simulating with no lapse
        for n in range(1, periodRemain + 1):
            FV_table[:, n] = np.maximum(FV_table[:, n - 1] - I_table[:, n - 1], 0) * (S_table[:, n] / S_table[:, n - 1]) * (1 - eta_g)
            GV_table[:, n] = np.maximum(FV_table[:, n], GV_table[:, n - 1])
            I_table[:, n] = GV_table[:, n] * withdrawal_rate

            mask = (I_table[:, n - 1] < FV_table[:, n - 1])
            dFV_table[:, n] = mask * (dFV_table[:, n - 1] - dI_table[:, n - 1]) \
                            * (S_table[:, n] / (S_table[:, n - 1])) * (1 - eta_g)
            flag_guarantee = (GV_table[:, n - 1] >= FV_table[:, n])
            dGV_table[:, n] = (1 - flag_guarantee) * dFV_table[:, n] + flag_guarantee * dGV_table[:, n - 1]
            dI_table[:, n] = withdrawal_rate * dGV_table[:, n]
            
    discount_rates = np.exp(-r * np.arange(1, periodRemain + 1))
    delta_vector = np.sum(discount_rates * (I_table[:, 1:] > FV_table[:, 1:])
                        * (dI_table[:, 1:] - dFV_table[:, 1:])
                        - dFV_table[:, 1:] * eta_n, axis=1) * survival_rate_outer

    delta = np.mean(delta_vector)
    delta_se = np.std(delta_vector) / np.sqrt(N)

    return delta, delta_se


def BS_outer(M, S_0, drift, volatility, GV_0, n_steps,
             withdrawal_rate, eta_g,
             lapse_table, lapse_flag, dynamic_flag,
             seed_diffusion):
    """
    :param M: # outer paths
    :param S_0: asset value at time 0
    :param drift: asset drift (per period)
    :param volatility: asset volatility (per period)
    :param GV_0: guarantee value for the VA contract
    :param n_steps: total time of VA contract
    :param seed_diffusion: random seed for the diffusion in RS model
    :return:    states: RS states for the outer simulation
                S_table: the underlying asset values
                FV_table: the fund values
                GV_table: the guaranteed values

    """

    # generate outer path of the asset
    rng = default_rng(seed_diffusion)
    Z = rng.normal(0, 1, [M, n_steps])
    S_table = np.zeros([M, n_steps + 1])
    S_table[:, 0] = S_0
    for n in range(n_steps):
        S_table[:, n + 1] = S_table[:, n] * np.exp(drift + volatility * Z[:, n])

    FV_table = S_table
    GV_table = np.full_like(FV_table, GV_0)
    I_table = GV_table * withdrawal_rate

    for j in range(S_table.shape[1] - 1):
        FV_table[:, j + 1] = np.maximum(FV_table[:, j] - I_table[:, j], 0) * (S_table[:, j + 1] / S_table[:, j]) * (1 - eta_g)
        GV_table[:, j + 1] = np.maximum(FV_table[:, j + 1], GV_table[:, j])
        I_table[:, j + 1] = GV_table[:, j + 1] * withdrawal_rate

    if lapse_flag:
        if dynamic_flag:
            dynamic_lapse_multiplier = np.concatenate([np.ones([M, 1]),
                                                       lapse.cal_dynamicLapseMultiplier(FV_table[:, :-1],
                                                                                        GV_table[:, :-1])],
                                                      axis=1)

            lapse_rate = dynamic_lapse_multiplier * lapse_table
        else:
            lapse_rate = lapse_table

        survival_rates = np.cumprod(1 - lapse_rate, axis=1)
    else:
        survival_rates = 1

    FV_table = FV_table * survival_rates
    GV_table = GV_table * survival_rates
    I_table = I_table * survival_rates

    return S_table, FV_table, GV_table, I_table, survival_rates


def BS_inner(N, stockValue, fundValue, guaranteeValue,
             r, volatility, periodRemain,
             withdrawal_rate, eta_g, eta_n,
             base_lapse, lapse_flag, dynamic_flag,
             seed_diffusion):

    if fundValue < withdrawal_rate * guaranteeValue:
        delta = 0
        delta_se = 0

    elif periodRemain > 0:

        rng = default_rng(seed_diffusion)
        Z = rng.normal(0, 1, [N, periodRemain])
        S_table = np.zeros([N, periodRemain + 1])
        S_table[:, 0] = stockValue
        for n in range(periodRemain):
            S_table[:, n + 1] = S_table[:, n] * np.exp(r + volatility * Z[:, n])

        FV_table = S_table
        GV_table = np.full_like(FV_table, guaranteeValue)
        I_table = np.zeros_like(GV_table)

        if lapse_flag:
            if dynamic_flag:
                dynamic_lapse_multiplier = lapse.cal_dynamicLapseMultiplier(FV_table[:, :-1], GV_table[:, :-1])
            else:
                dynamic_lapse_multiplier = np.ones_like(FV_table[:, :-1])

            inner_baseLapse = base_lapse[-periodRemain:].reshape(1, -1)
            lapse_rate = dynamic_lapse_multiplier * inner_baseLapse
            survival_rates = 1 - lapse_rate
            cumulative_rates = np.cumprod(survival_rates, axis=1)
        else:
            survival_rates = 1
            cumulative_rates = 1

        # Calculate deltas and standard errors
        dFV_table = FV_table / (S_table)
        dGV_table = np.zeros_like(dFV_table)
        dI_table = np.zeros_like(dFV_table)

        for n in range(1, periodRemain + 1):
            FV_table[:, n] = np.maximum(FV_table[:, n - 1] - I_table[:, n - 1], 0) * (S_table[:, n] / S_table[:, n - 1]) * (1 - eta_g)
            GV_table[:, n] = np.maximum(FV_table[:, n], GV_table[:, n - 1])
            I_table[:, n] = GV_table[:, n] * withdrawal_rate

            mask = (I_table[:, n - 1] < FV_table[:, n - 1])
            dFV_table[:, n] = mask * (dFV_table[:, n - 1] - dI_table[:, n - 1]) \
                            * (S_table[:, n] / (S_table[:, n - 1])) * (1 - eta_g)
            flag_guarantee = (GV_table[:, n - 1] >= FV_table[:, n])
            dGV_table[:, n] = (1 - flag_guarantee) * dFV_table[:, n] + flag_guarantee * dGV_table[:, n - 1]
            dI_table[:, n] = withdrawal_rate * dGV_table[:, n]

        discount_rates = np.exp(-r * np.arange(1, periodRemain + 1))
        delta_vector = np.sum(discount_rates * (I_table[:, 1:] > FV_table[:, 1:])
                            * (dI_table[:, 1:] - dFV_table[:, 1:])
                            - dFV_table[:, 1:] * eta_n, axis=1)

        delta = np.mean(delta_vector)
        delta_se = np.std(delta_vector) / np.sqrt(N)

    else:

        delta = 0
        delta_se = 0

    return delta, delta_se
