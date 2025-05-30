import numpy as np
from numpy.random import default_rng
from scipy.stats import norm

import lapse


def RS_transition(state_before, transition_prob, seed):
    rng = default_rng(seed)

    # calculates the states for regime-switching models
    num = state_before.shape[0]
    rand = rng.uniform(0, 1, num)

    transition_vector = np.where(state_before == 0, transition_prob[0], transition_prob[1])
    transition_flag = (rand <= transition_vector)

    # if transition, state = (state+1) mod 2
    state_after = (state_before + transition_flag) % 2

    return state_after


def RS_outer(M, S_0, drift, volatility, GV_0, initial_state, transition_prob, n_steps,
             lapse_table, lapse_flag, dynamic_flag,
             seed_states, seed_diffusion):
    """
    :param M: # outer paths
    :param S_0: asset value at time 0
    :param drift: asset drift (per period)
    :param volatility: asset volatility (per period)
    :param GV_0: guarantee value for the VA contract
    :param initial_state: the initial RS state
    :param transition_prob: transition probability vector
    :param n_steps: total time of VA contract
    :param lapse_table: base lapse table, to be multiplied by the dynamic lapse multiplier
    :param seed_states: random seed for RS states
    :param seed_diffusion: random seed for the diffusion in RS model
    :return:    states: RS states for the outer simulation
                S_table: the underlying asset values
                FV_table: the fund values
                GV_table: the guaranteed values

    """

    # generate the RS states for outer simulation
    states = np.zeros([M, n_steps + 1], dtype=int)
    states[:, 0] = initial_state
    for n in range(n_steps):
        states[:, n + 1] = RS_transition(states[:, n], transition_prob, seed_states)

    # set drift and diffusion according to regime, the last period (maturity) requires no drift and volatility
    drift_vector = np.where(states[:, :n_steps] == 0, drift[0], drift[1])
    volatility_vector = np.where(states[:, :n_steps] == 0, volatility[0], volatility[1])

    # generate outer path of the asset
    rng = default_rng(seed_diffusion)
    Z = rng.normal(0, 1, [M, n_steps])
    S_table = np.zeros([M, n_steps + 1])
    S_table[:, 0] = S_0
    for n in range(n_steps):
        S_table[:, n + 1] = S_table[:, n] * np.exp(drift_vector[:, n] + volatility_vector[:, n] * Z[:, n])

    FV_table = S_table
    GV_table = np.full_like(FV_table, GV_0)

    if lapse_flag:
        if dynamic_flag:
            dynamic_lapse_multiplier = np.concatenate([np.ones([M, 1]),
                                                       lapse.cal_dynamicLapseMultiplier(S_table[:, :-1],
                                                                                        GV_table[:, :-1])],
                                                      axis=1)

            lapse_rate = dynamic_lapse_multiplier * lapse_table
        else:
            lapse_rate = lapse_table

        survival_rates = np.cumprod(1 - lapse_rate, axis=1)
    else:
        survival_rates = np.ones_like(FV_table)

    FV_table = FV_table * survival_rates
    GV_table = GV_table * survival_rates

    return states, S_table, FV_table, GV_table, survival_rates


def RS_inner(N, fundValue, guaranteeValue, state, survivalRate_outer,
             r, drift, volatility, periodRemain,
             base_lapse, lapse_flag, dynamic_flag,
             transition_prob, seed_states, seed_diffusion):


    price = np.maximum(guaranteeValue - fundValue, 0)
    delta = -1 * (fundValue < guaranteeValue)
    price_se = 0
    delta_se = 0

    if periodRemain > 0:

        states = np.zeros([N, periodRemain + 1], dtype=int)
        states[:, 0] = state
        for n in range(periodRemain):
            states[:, n + 1] = RS_transition(states[:, n], transition_prob, seed_states)

        drift_vector = np.where(states[:, :periodRemain] == 0, drift[0], drift[1])
        volatility_vector = np.where(states[:, :periodRemain] == 0, volatility[0], volatility[1])

        rng = default_rng(seed_diffusion)
        Z = rng.normal(0, 1, [N, periodRemain])
        logReturn = drift_vector + volatility_vector * Z

        cumulativeReturn = np.concatenate([np.ones([N, 1]),
                                           np.exp(np.cumsum(logReturn, axis=1))], axis=1)


        fundValues = fundValue * cumulativeReturn

        if lapse_flag:
            if dynamic_flag:
                dynamic_lapse_multiplier = lapse.cal_dynamicLapseMultiplier(fundValues[:, :-1], guaranteeValue)
            else:
                dynamic_lapse_multiplier = np.ones_like(fundValues[:, :-1])

            inner_baseLapse = base_lapse[-periodRemain:].reshape(1, -1)
            lapse_rate = dynamic_lapse_multiplier * inner_baseLapse
            survivalRate_inner = np.prod(1 - lapse_rate, axis=1)
        else:
            survivalRate_inner = 1

        flag_guarantee = fundValues[:, -1] < guaranteeValue
            
        # calculate payoff, price, and delta
        payoff_vector = np.exp(-r * periodRemain) \
                        * flag_guarantee * (guaranteeValue - fundValues[:, -1]) * survivalRate_inner

        delta_vector = np.exp(-r * periodRemain) * flag_guarantee \
                    * (-cumulativeReturn[:, -1]) * survivalRate_inner * survivalRate_outer

        price = np.mean(payoff_vector)
        price_se = np.std(payoff_vector) / np.sqrt(N)
        delta = np.mean(delta_vector)
        delta_se = np.std(delta_vector) / np.sqrt(N)

    return price, delta, price_se, delta_se


def BS_outer(M, S_0, drift, volatility, GV_0, n_steps, lapse_table, lapse_flag, seed_diffusion):
    """
    :param M: # outer paths
    :param S_0: asset value at time 0
    :param drift: asset drift (per period)
    :param volatility: asset volatility (per period)
    :param GV_0: guarantee value for the VA contract
    :param n_steps: total time of VA contract
    :param lapse_table: base lapse table, to be multiplied by the dynamic lapse multiplier
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

    survival_rates = np.ones_like(FV_table)

    if lapse_flag:
        survival_rates = np.cumprod(1 - lapse_table, axis=1)

        FV_table = FV_table * survival_rates
        GV_table = GV_table * survival_rates

    return S_table, FV_table, GV_table, survival_rates


def BS_inner(fundValue, guaranteeValue, survivalRate_outer,
             r, volatility, periodRemain, base_lapse, lapse_flag):

    price = np.maximum(guaranteeValue - fundValue, 0)
    delta = -1 * (fundValue < guaranteeValue)

    if periodRemain > 0:

        d1 = (np.log(fundValue / guaranteeValue) + r + volatility ** 2 / 2) / (volatility * np.sqrt(periodRemain))
        d2 = d1 - volatility * np.sqrt(periodRemain)

        if lapse_flag:
            survivalRate_inner = np.prod(1 - base_lapse[-periodRemain:])
        else:
            survivalRate_inner = 1

        price = guaranteeValue * np.exp(-r * periodRemain) * norm.cdf(-d2) \
                - fundValue * norm.cdf(-d1) * survivalRate_inner
        delta = - norm.cdf(-d1) * survivalRate_inner * survivalRate_outer

    return price, delta
