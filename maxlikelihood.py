import random
import sys
import math
import numpy as np
import pandas as pd
from pandas.core.base import NoNewAttributesMixin
import matplotlib.pyplot as plt
import time
import constant
import constant

def main():
    filenames = ['small2', 'medium2', 'large2']
    for filename in filenames:
        compute(filename+'.csv', filename+'_maxlikelihood.policy')
    pass

def compute(infile, outfile):
    start = time.time()
    D = pd.read_csv(infile)
    T, R = generateModel(D)
    
    discount = constant.DISCOUNT
    policy = value_iteration(R,T,discount)
    np.savetxt(outfile,policy)
    taken = time.time() - start
    print("time taken for ",infile," = ",taken," seconds")
    

def generateModel(data):
    R = np.zeros((constant.STATE_SIZE,constant.ACTION_SIZE))
    N = np.zeros((constant.STATE_SIZE,constant.ACTION_SIZE,constant.STATE_SIZE))
    for index, row in data.iterrows():
        s = row['s']
        a = row['a']
        s_prime = row['sp']
        reward = row['r']
        N[s,a,s_prime] += 1
        R[s,a] += reward
    freq = np.sum(N,axis=2)
    for s in range(constant.STATE_SIZE):
        for a in range(constant.ACTION_SIZE):
            if (freq[s,a] == 0):
                N[s,a,:] = 0
                R[s,a] = 0
            else:
                N[s,a,:] /= freq[s,a]
                R[s,a] /= freq[s,a]
    return N, R


def value_iteration(R, T, discount):
    iterations = 100
    U = np.zeros(constant.STATE_SIZE)
    for _ in range(iterations):
        changed = np.zeros(constant.STATE_SIZE)
        for s in range(constant.STATE_SIZE):
            utility = 0
            for a in range(constant.ACTION_SIZE):
                reward = R[s,a]
                transition = T[s,a,:]
                transition.reshape(constant.STATE_SIZE)
                candidate = R[s,a] + discount*np.dot(transition,U)
                utility = max(candidate,utility)
            changed[s] = utility
        U = changed
    policy = np.zeros(constant.STATE_SIZE)
    for s in range(constant.STATE_SIZE):
        chance = np.zeros(constant.ACTION_SIZE)
        for a in range(constant.ACTION_SIZE):
            reward = R[s,a]
            transition = T[s,a,:]
            transition.reshape(constant.STATE_SIZE)
            chance[a] = R[s,a] + discount*np.dot(transition,U)
        policy[s] = np.argmax(chance)
    return policy



if __name__ == '__main__':
    main()