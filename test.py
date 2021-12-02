import numpy as np
import pandas as pd
import math
from scipy.stats import truncnorm
import constant

SIM_SIZE = constant.SIM_SIZE
AVG_V = constant.AVG_V
VAR_V = constant.VAR_V
VAR_X = constant.VAR_X

def generateTest(num_objects, filename):
    samples = []
    for i in range(num_objects):
        obj_pos = math.floor(truncnorm.rvs(-SIM_SIZE/(VAR_X*2), SIM_SIZE/(VAR_X*2), loc=SIM_SIZE / 2, scale=VAR_X))
        d = np.random.choice([-1, 1])
        obj_vel = d * math.floor(truncnorm.rvs(-AVG_V/VAR_V, (constant.MAX_V-AVG_V)/VAR_V, loc=AVG_V, scale=VAR_V))
        max_t = np.random.randint(constant.MIN_T, constant.MAX_T + 1)
        samples.append([obj_pos, obj_vel, max_t])
    df = pd.DataFrame(samples, columns=['obj_pos', 'obj_vel', 'max_t'])
    df.to_csv(filename, index=False)


def runTest(test_file, policy_file):
    score = 0
    data = pd.read_csv(test_file)
    rows, _ = data.shape
    policy = pd.read_csv(policy_file, header=None)[0]
    for i in range(rows):
        # Define initial states
        obj_pos = data.obj_pos[i]
        agent_pos = SIM_SIZE // 2
        obj_vel = data.obj_vel[i]
        agent_vel = int(0)
        max_t = data.max_t[i]
        # Loop through time steps
        for t in range(max_t):
            s = s_to_idx(agent_pos - obj_pos, agent_vel - obj_vel)
            # In case of collision
            if obj_pos == agent_pos and t == max_t-1:
                score += constant.COLLISION_COST
                break
            action = policy[s]
            if action == 1:
                if agent_vel < constant.MAX_V:
                    agent_vel += 1
                score += constant.FUEL_COST
            elif action == 2:
                if agent_vel > -constant.MAX_V:
                    agent_vel -= 1
                score += constant.FUEL_COST
            agent_pos = agent_pos + agent_vel
            obj_pos = obj_pos + int(obj_vel)
            if agent_pos >= SIM_SIZE or agent_pos < 0 or obj_pos >= SIM_SIZE or obj_pos < 0:
                break
    print("TEST COMPLETE")
    print("Score with " + policy_file + " = " + str(score))

'''
Convert state tuple to an integer index
sep is in range [-SIM_SIZE + 1, SIM_SIZE - 1]
vel is in range [-MAX_V * 2, MAX_V * 2]
'''
def s_to_idx(sep, vel):
    n = sep + SIM_SIZE - 1
    return (n * constant.NUM_V) + vel + (constant.MAX_V * 2)

def main():
    #generateTest(10000, 'test.csv')
    files = ['small.policy', 'medium.policy', 'large.policy',
             'small2.policy', 'medium2.policy', 'large2.policy']
    for file in files:
        runTest('test.csv', file)

if __name__ == '__main__':
    main()