from gurobipy import GRB, Model, quicksum
import numpy as np
from problem_dynamics import build_dynamics_fifo
from lp_solver import lp_solver
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    life_time = 2
    lead_time = 1
    unit_lost_cost = 5
    unit_hold_cost = 1
    unit_perish_cost = 7
    unit_order_cost = 3
    max_order = 5
    mean_demand = 2
    cv = 0.5

    parameters = {
        'life_time': life_time,
        'lead_time': lead_time,
        'unit_lost_cost': unit_lost_cost,
        'unit_hold_cost': unit_hold_cost,
        'unit_perish_cost': unit_perish_cost,
        'unit_order_cost': unit_order_cost,
        'max_order': max_order,
        'mean_demand': mean_demand,
        'cv': cv
    }

    dynamics, state_space, action_space, value, policy = build_dynamics_fifo(
        parameters)

    # set parameters for LP
    gamma = 0.99

    # calculate r(s,a) and p(s'|s,a)
    r = np.zeros((len(state_space), len(action_space)))
    p = np.zeros((len(state_space), len(action_space), len(state_space)))

    # transfer state to index
    state_to_index = {}
    for i, state in enumerate(state_space):
        state_to_index[state] = i

    # calculate r(s,a) and p(s'|s,a)
    for state in state_space:
        for action in action_space:
            for (next_state, reward), prob in dynamics[state, action].items():
                r[state_to_index[state], action] += reward * prob
                p[state_to_index[state], action,
                    state_to_index[next_state]] += prob
                
    # assert sum of p(s'|s,a) = 1
    for state in state_space:
        for action in action_space:
            assert abs(sum(p[state_to_index[state], action, :]) - 1) < 1e-6


    # solve LP
    model = lp_solver(r, p, gamma)

    # state value
    for state in state_space:
        value[state] = model.getVarByName(
            'v_{}'.format(state_to_index[state])).x

    for state in state_space:
        best_value = -np.inf
        for action in action_space:
            value_temp = sum([prob * (reward + gamma * value[next_state])
                             for (next_state, reward), prob in dynamics[state, action].items()])
            if value_temp > best_value:
                best_value = value_temp
                policy[state] = action


    # plot policy heatmap for each state [i,:,:]
    for i in range(max_order+1):
        sns.heatmap(policy[i, :, :], cmap='YlGnBu',
                    vmin=0, vmax=np.max(policy), annot=True, fmt='d')
        # invert y axis
        plt.gca().invert_yaxis()
        # show action value
        plt.title('Policy for state {}'.format(i))
        plt.xlabel("age 2")
        plt.ylabel("age 1")
        plt.show()

    # plot value heatmap for each state [i,:,:]
    for i in range(max_order+1):
        sns.heatmap(value[i, :, :], cmap='YlGnBu')
        plt.title('Value for state {}'.format(i))
        plt.show()