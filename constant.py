# CONSTANTS FOR QLEARN.PY
# SHARED hyperparameters
SIM_SIZE = 40         # number of positions for object / agent
MAX_V = 8              # maximum magnitude of velocity that can be achieved by agent or object

# SIMULATION hyperparameters
FUEL_COST = -2          # cost of changing agent velocity
COLLISION_COST = -5000  # cost of collision
MIN_T = 10              # min time steps to collision
MAX_T = 20             # max time steps of simulation
AVG_V = 2               # mean object velocity
VAR_V = 2               # std of object velocity
VAR_X = 8             # std of initial object position

# QLEARN hyperparameters
NUM_ITERS = 20
DISCOUNT = 0.95
LR = 0.1

# Useful constants
NUM_V = (MAX_V * 4) + 1     # number of possible values for difference in velocity
NUM_P = (SIM_SIZE * 2) - 1  # number of possible values for difference in position
STATE_SIZE = NUM_P * NUM_V  # number of possible states
ACTION_SIZE = 3             # number of possible actions
