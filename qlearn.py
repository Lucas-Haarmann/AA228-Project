import random
import sys
import math
import numpy as np
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
import matplotlib.pyplot as plt
import time

def Q_learning (data, Q, discount, lr, iterations):
    for _ in range(iterations):
        for index, row in data.iterrows():
            s = row['s']
            a = row['a']
            s_prime = row['sp']
            reward = row['r']
            curr = Q[s,a]
            Q[s,a] = curr + lr * (reward + discount * np.max(Q[s_prime,:]) - curr)
    policy = np.argmax(Q, axis = 1)
    return policy

def compute(infile, outfile):
    start = time.time()
    D = pd.read_csv(infile)
    S, A, discount, lr, iterations = 0, 2, 0.95, 1e-1, 2
    if (infile == 'small.csv'):
        S, iterations = 100, 2
    elif (infile == 'medium.csv'):
        S, iterations = 10000, 2
    else:
        S, iterations = 100000, 35
    Q = np.zeros((S,A))
    policy = Q_learning(D, Q, discount, lr, iterations) + 1
    np.savetxt(outfile,policy)
    taken = time.time() - start
    print("time taken for ",infile," = ",taken," seconds")


def main():
    filenames = ['small', 'medium', 'large']
    for filename in filenames:
        compute(filename+'.csv', filename+'.policy')

if __name__ == '__main__':
    main()