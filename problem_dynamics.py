# this code builds the dynamics of the single echelon perishable inventory problem
# value iteration is used to solve the problem

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gamma
from itertools import product


def build_dynamics_fifo(parameters):

    life_time = parameters['life_time']
    lead_time = parameters['lead_time']
    unit_lost_cost = parameters['unit_lost_cost']
    unit_hold_cost = parameters['unit_hold_cost']
    unit_perish_cost = parameters['unit_perish_cost']
    unit_order_cost = parameters['unit_order_cost']
    max_order = parameters['max_order']
    mean_demand = parameters['mean_demand']
    cv = parameters['cv']

    # parameter for gamma distribution
    EPSILON = 1e-4

    # Calculate variance
    variance = (cv * mean_demand) ** 2
    # Calculate shape and scale parameters
    shape = (mean_demand ** 2) / variance
    scale = variance / mean_demand

    # find the maximum demand
    max_demand = int(gamma.ppf(1-EPSILON, shape, scale=scale))

    # build state space, state contains life_time+lead_time elements, each element is up to max_order
    state_space = []
    element_values = list(range(max_order + 1))
    state_elements = [element_values] * (life_time + lead_time)

    for state in product(*state_elements):
        state_space.append(state)

    # state space size should be (max_order+1)^(life_time+lead_time)
    # print('state space size: ', len(state_space))

    # build action space, action is the order quantity, from 0 to max_order
    action_space = list(range(max_order + 1))

    # build probability dictionary for each possible integer demand
    demand_prob = {}
    # transform gamma distribution to discrete distribution, round to nearest integer
    for demand in range(max_demand + 1):
        if demand == 0:
            demand_prob[demand] = gamma.cdf(0.5, shape, scale=scale)
        elif demand == max_demand:
            demand_prob[demand] = 1 - \
                gamma.cdf(max_demand - 0.5, shape, scale=scale)
        else:
            demand_prob[demand] = gamma.cdf(
                demand + 0.5, shape, scale=scale) - gamma.cdf(demand - 0.5, shape, scale=scale)

    assert sum(demand_prob.values()
               ) == 1, 'demand probability does not sum to 1'

    # build dynamics
    dynamics = {}
    for state in state_space:
        for action in action_space:
            dynamics[state, action] = {}
            for demand in range(max_demand + 1):
                # calculate next state probability
                prob = demand_prob[demand]

                # satisfy demand following FIFO policy
                state_temp = list(state)
                for i in range(life_time):
                    if state_temp[-i-1] >= demand:
                        state_temp[-i-1] -= demand
                        demand = 0
                        break
                    else:
                        demand -= state_temp[-i-1]
                        state_temp[-i-1] = 0

                # update state
                next_state = [0] * (life_time + lead_time)
                for i in range(life_time + lead_time):
                    if i == 0:
                        next_state[i] = action
                    else:
                        next_state[i] = state_temp[i - 1]
                next_state = tuple(next_state)

                # calculate reward
                order_cost = unit_order_cost * action
                perished_cost = unit_perish_cost * state_temp[-1]
                lost_cost = unit_lost_cost * demand
                hold_cost = unit_hold_cost * sum(next_state[lead_time:])

                reward = - (order_cost + perished_cost + lost_cost + hold_cost)

                # update dynamics
                if (next_state, reward) in dynamics[state, action]:
                    dynamics[state, action][next_state, reward] += prob
                else:
                    dynamics[state, action][next_state, reward] = prob
    # build initial value and policy
    policy_shape = tuple([max_order + 1] * (life_time + lead_time))
    init_value = np.zeros(policy_shape)
    init_policy = np.zeros(policy_shape, dtype=int)

    return dynamics, state_space, action_space, init_value, init_policy