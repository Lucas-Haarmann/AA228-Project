import sys
import time
import numpy as np
import pandas as pd

FUEL_COST = -5
COLLISION_COST = -500
MAX_T = 100

def generateSamples(num_objects):
    samples = []
    for i in range(num_objects):
        # Define initial states
        obj_pos = np.random.randint(0, 200)
        agent_pos = np.random.randint(0, 200)
        dir = np.random.choice([-1, 1])
        obj_vel = int(dir * np.random.normal(3))
        agent_vel = int(0)
        max_t = np.random.randint(50, MAX_T + 1)
        # Loop through time steps
        for t in range(max_t):
            row = [0, 0, 0, 0, 0, 0] # state, state, action, reward, s primed, sprimed
            dir = agent_vel - obj_vel
            row[0] = agent_pos - obj_pos
            row[1] = dir
            if obj_pos == agent_pos and t == max_t-1:
                row[3] = COLLISION_COST
            action = np.random.randint(0, 3) # STAY, ACC UP, ACC DOWN
            row[2] = action
            if action == 1:
                agent_vel += 1
                row[3] = FUEL_COST
            elif action == 2:
                agent_vel -= 1
                row[3] = FUEL_COST
            agent_pos = agent_pos + agent_vel
            obj_pos = obj_pos + int(obj_vel)
            if agent_pos >= 200 or agent_pos < 0 or obj_pos >= 200 or obj_pos < 0:
                samples.append(row)
                break
            row[4] = agent_pos - obj_pos
            row[5] = agent_vel - obj_vel
            samples.append(row)
    df = pd.DataFrame(samples, columns=['sep', 'vel', 'a', 'r', 'sep_p', 'vel_p'])
    df.to_csv('large.csv', index=False)





def main():
    generateSamples(100000)

if __name__ == '__main__':
    main()