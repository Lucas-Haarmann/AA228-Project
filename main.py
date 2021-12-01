import numpy as np
import pandas as pd
import math
from scipy.stats import truncnorm

# Hyperparameters to change (update qlearn.py too)
FUEL_COST = -2          # cost of changing agent velocity
COLLISION_COST = -1000  # cost of collision
SIM_SIZE = 40         # number of positions for object / agent
MIN_T = 10
MAX_T = 20             # max time steps of simulation
MAX_V = 8              # maximum magnitude of velocity that can be achieved by agent or object
AVG_V = 2               # mean object velocity
VAR_V = 2               # std of object velocity
VAR_X = 8             # std of initial object position

# Useful constants
NUM_V = (MAX_V * 4) + 1     # number of possible values for difference in velocity
NUM_P = (SIM_SIZE * 2) - 1  # number of possible values for difference in position
STATE_SIZE = NUM_P * NUM_V  # number of possible states

def generateSamples(num_objects, filename):
    samples = []
    for i in range(num_objects):
        # Define initial states
        obj_pos = math.floor(truncnorm.rvs(-SIM_SIZE/(VAR_X*2), SIM_SIZE/(VAR_X*2), loc=SIM_SIZE / 2, scale=VAR_X))
        agent_pos = SIM_SIZE // 2
        d = np.random.choice([-1, 1])
        obj_vel = d * math.floor(truncnorm.rvs(-AVG_V/VAR_V, (MAX_V-AVG_V)/VAR_V, loc=AVG_V, scale=VAR_V))
        agent_vel = int(0)
        max_t = np.random.randint(MIN_T, MAX_T + 1)
        # Loop through time steps
        for t in range(max_t):
            row = [0, 0, 0, 0] # state, action, reward, new state
            row[0] = s_to_idx(agent_pos - obj_pos, agent_vel - obj_vel)
            # In case of collision
            if obj_pos == agent_pos and t == max_t-1:
                row[2] = COLLISION_COST
                samples.append(row)
                break
            action = np.random.choice([0, 1, 2],
                                      p=[0.5, 0.25, 0.25])  # STAY, ACC UP, ACC DOWN
            row[1] = action
            if action == 1:
                if agent_vel < MAX_V:
                    agent_vel += 1
                row[2] = FUEL_COST
            elif action == 2:
                if agent_vel > -MAX_V:
                    agent_vel -= 1
                row[2] = FUEL_COST
            agent_pos = agent_pos + agent_vel
            obj_pos = obj_pos + int(obj_vel)
            if agent_pos >= SIM_SIZE or agent_pos < 0 or obj_pos >= SIM_SIZE or obj_pos < 0:
                samples.append(row)
                break
            row[3] = s_to_idx(agent_pos - obj_pos, agent_vel - obj_vel)
            samples.append(row)
    df = pd.DataFrame(samples, columns=['s', 'a', 'r', 'sp'])
    df.to_csv(filename, index=False)

'''
Convert state tuple to an integer index
sep is in range [-SIM_SIZE + 1, SIM_SIZE - 1]
vel is in range [-MAX_V * 2, MAX_V * 2]
'''
def s_to_idx(sep, vel):
    n = sep + SIM_SIZE - 1
    return (n * NUM_V) + vel + (MAX_V * 2)

def main():
    files = [('small', 100), ('medium', 10000), ('large', 50000)]
    for name, samples in files:
        generateSamples(samples, name+'.csv')

if __name__ == '__main__':
    main()