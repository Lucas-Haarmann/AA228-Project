import numpy as np
import pandas as pd
import time
import constant

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
    Q = np.zeros((constant.STATE_SIZE, 3))
    policy = Q_learning(D, Q, constant.DISCOUNT, constant.LR, constant.NUM_ITERS) + 1
    np.savetxt(outfile,policy)
    taken = time.time() - start
    print("time taken for ",infile," = ",taken," seconds")


def main():
    filenames = ['small2', 'medium2', 'large2']
    for filename in filenames:
        compute(filename+'.csv', filename+'.policy')

if __name__ == '__main__':
    main()