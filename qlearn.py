import random
import sys
import math
import numpy as np
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
import matplotlib.pyplot as plt
import time

# Hyperparameters to change (check main.py)
SIM_SIZE = 40             # no of agent / object position values in range [0, SIM_SIZE-1]
MAX_V = 8                  # maximum possible velocity value

NUM_ITERS = 10
DISCOUNT = 0.95
LR = 0.1

NUM_V = (MAX_V * 4) + 1     # number of possible values for difference in velocity
NUM_P = (SIM_SIZE * 2) - 1  # number of possible values for difference in position
STATE_SIZE = NUM_P * NUM_V  # number of possible states

def Q_learning (data, Q, discount, lr, iterations):
    for _ in range(iterations):
        for index, row in data.iterrows():
            s = row['s']
            a = row['a']
            s_prime = row['sp']
            reward = row['r']
            curr = Q[s,a]
            Q[s,a] = curr + lr * (reward + discount * np.max(Q[s_prime,:]) - curr)
    policy = np.argmax(Q, axis=1).astype(int)
    return policy


def compute(infile, outfile):
    start = time.time()
    D = pd.read_csv(infile)
    Q = np.zeros((STATE_SIZE, 3))
    policy = Q_learning(D, Q, DISCOUNT, LR, NUM_ITERS) + 1
    np.savetxt(outfile,policy)
    taken = time.time() - start
    print("time taken for ",infile," = ",taken," seconds")


def main():
    filenames = ['small', 'medium', 'large']
    for filename in filenames:
        compute(filename+'.csv', filename+'.policy')

if __name__ == '__main__':
    main()