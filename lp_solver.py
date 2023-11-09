'''
linear programming solver for MDPs.

r: reward matrix, n_state * n_action
p: transition probability matrix, n_state * n_action * n_state
gamma: discount factor
'''

from gurobipy import GRB, Model, quicksum

def lp_solver(r, p, gamma):

    action_set = set(range(r.shape[1]))
    state_set = set(range(r.shape[0]))
    n_state = len(state_set)

    # create a model instance
    model = Model()

    # create variables
    for s in range(n_state):
        model.addVar(name=f'v_{s}', lb=-GRB.INFINITY)
    
    # update the model
    model.update()

    # create constraints
    for state in state_set:
        for action in action_set:
            model.addConstr(model.getVarByName(f'v_{state}') >= quicksum(
                gamma * p[state, action, next_state] * model.getVarByName(f'v_{next_state}') for next_state in state_set ) + r[state, action])

    # set objective
    model.setObjective(quicksum(model.getVarByName(
        f'v_{state}') for state in state_set ), GRB.MINIMIZE)

    # optimize
    model.optimize()

    return model