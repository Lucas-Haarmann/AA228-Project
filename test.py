import numpy as np
import pandas as pd
import math
from scipy.stats import truncnorm
import constant
import matplotlib.pyplot as plt

SIM_SIZE = constant.SIM_SIZE
AVG_V = constant.AVG_V
VAR_V = constant.VAR_V
VAR_X = constant.VAR_X

'''
Function to generate test file (only used once to generate test.csv)
'''
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

'''
Function to simulate failures and plot graph
'''
def simulate(policy_file):
    collision = False
    policy = pd.read_csv(policy_file, header=None)[0]
    while not collision:
        agent = []
        obj = []
        obj_pos = math.floor(truncnorm.rvs(-SIM_SIZE/(VAR_X*2), SIM_SIZE/(VAR_X*2), loc=SIM_SIZE / 2, scale=VAR_X))
        d = np.random.choice([-1, 1])
        obj_vel = d * math.floor(truncnorm.rvs(-AVG_V/VAR_V, (constant.MAX_V-AVG_V)/VAR_V, loc=AVG_V, scale=VAR_V))
        max_t = np.random.randint(constant.MIN_T, constant.MAX_T + 1)
        agent_pos = SIM_SIZE // 2
        agent_vel = int(0)
        score = 0
        for t in range(max_t):
            agent.append(agent_pos)
            obj.append(obj_pos)
            s = s_to_idx(agent_pos - obj_pos, agent_vel - obj_vel)
            # In case of collision
            if obj_pos == agent_pos and t == max_t-1:
                collision = True
                score += constant.COLLISION_COST
                break
            action = policy[s] - 1
            if action == 1:
                if agent_vel < constant.MAX_V:
                    agent_vel += 1
                score += constant.FUEL_COST
            elif action == 2:
                acted = True
                if agent_vel > -constant.MAX_V:
                    agent_vel -= 1
                score += constant.FUEL_COST
            agent_pos = agent_pos + agent_vel
            obj_pos = obj_pos + int(obj_vel)
            if agent_pos >= SIM_SIZE or agent_pos < 0 or obj_pos >= SIM_SIZE or obj_pos < 0:
                break
    print(score)
    x = np.arange(0, len(agent), 1)
    plt.plot(x, agent, label='Agent')
    plt.plot(x, obj, label='Object')
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    plt.legend()
    plt.show()


'''
Run policy on test file and compute score
'''
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
            action = policy[s] - 1
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
    #simulate('medium.policy')
    files = ['small2_maxlikelihood.policy', 'medium2_maxlikelihood.policy', 'large2_maxlikelihood.policy']
    for file in files:
        runTest('test.csv', file)

if __name__ == '__main__':
    main()