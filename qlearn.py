from pandas import read_csv
from random import randint
import constant as const
import sys
import timeit
import numpy as np

def load_data():
    """
    LOAD DATA:
    Uses command line argument to select and load the correct data file. Valid command line arguments are "large.csv", 
    "medium.csv", and "small.csv". If an invalid argument is given, defaults to "small.csv". If no arguments are given,
    raises system error.
    Return: pandas dataframe with data loaded 
    Return: gamma: (float) discount factor for loaded data
    Return: n_states: (int) size of state space
    Return: n_actions: (int) size of action space
    """
    try:
        arg = sys.argv[1]
    except IndexError:
        raise SystemExit("Please enter a valid argument: small.csv, medium.csv, or large.csv")

    if arg == "large.csv":
        data_fp = const.LARGE_FP
        gamma = const.LARGE_GAMMA
        n_states = const.LARGE_N_STATES
        n_actions = const.LARGE_N_ACTIONS
    elif arg == "medium.csv":
        data_fp = const.MEDIUM_FP
        gamma = const.MEDIUM_GAMMA
        n_states = const.MEDIUM_N_STATES
        n_actions = const.MEDIUM_N_ACTIONS
    else:
        data_fp = const.SMALL_FP
        gamma = const.SMALL_GAMMA
        n_states = const.SMALL_N_STATES
        n_actions = const.SMALL_N_ACTIONS
    
    return read_csv(data_fp), gamma, n_states, n_actions

def update(Q, s, a, r, sp, alpha, gamma):
    """
    UPDATE:
    Updates action value function Q given information about a state transition and its reward.
    Param: Q: (numpy array of size (n_states, n_actions)) action value function
    Param: s: (int) current state
    Param: a: (int) action
    Param: r: (float) state transition reward
    Param: sp: (int) next state
    Param: alpha: (float) learning rate
    Param: gamma: (float) discount factor
    Returns: implicit changes to Q
    """
    Q[s-1,a-1] += alpha * (r + gamma * max(Q[sp-1,:]) - Q[s-1,a-1])

def create_policy_file(Q, n_states, n_actions):
    """
    CREATE .POLICY FILE:
    Given the state action function array, creates and saves a .policy CSV-style file where the number of rows equals 
    the number of states and the number of columns equals the number of actions.
    Param: Q: (np array of size (n_states, n_actions)) action value function
    """
    if n_states == const.LARGE_N_STATES:
        filename = "large.policy"
    elif n_states == const.MEDIUM_N_STATES:
        filename = "medium.policy"
    else:
        filename = "small.policy"

    policy = np.zeros((n_states, 1), dtype=int)
    unexplored_states= np.where(~Q.any(axis=1))[0]  # Get row indices of rows that are all zeros (unexplored states)

    for i in range(n_states):
        if i in unexplored_states:  # If state is unexplored (i.e. the row is all zeros in Q), assign a random action to that state
            policy[i, 0] = randint(1, n_actions)
        else:
            policy[i, 0] = int(np.where(Q == max(Q[i,:]))[1][0] + 1)  # Get action number of best action

    np.savetxt(filename, policy, fmt="%i")


def main():
    
    # Load data and gamma
    data, gamma, n_states, n_actions = load_data()

    m = len(data.index)  # m = number of samples in dataset
    alpha = 1/m          # learning rate
    Q = np.zeros((n_states, n_actions), dtype=float)

    start_time = timeit.default_timer()
    for i in range(m):
        s = data.iloc[i]['s']
        a = data.iloc[i]['a']
        r = data.iloc[i]['r']
        sp = data.iloc[i]['sp']
        update(Q, s, a, r, sp, alpha, gamma)
    stop_time = timeit.default_timer()
    print("Training time: ", stop_time - start_time)

    create_policy_file(Q, n_states, n_actions)

if __name__ == '__main__':
    main()
