import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
from Environment import Environment
from ValueIteration import Agent

def runForV(envName):
    env = Environment(envName)
    agent = Agent(env, 4)
    agent.train()

    V = np.reshape(agent.V, (env.grid_size, env.grid_size)).astype(dtype=np.int)
    maxV = np.amax(V)
    minV = np.amin(V)

    V = (V - minV) / (maxV - minV)

    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i][j] == 'G':
                V[i][j] = -1.0
                break

    return V, np.reshape(agent.V, (env.grid_size, env.grid_size)), agent.V

def runForPolicy(envName, V):
    env = Environment(envName)

    actions = ["U", "L", "D", "R"]
    results = [["G" for j in range(env.grid_size)] for i in range(env.grid_size)]

    for S in range(len(V)):
        env.set_current_state(S)
        Q = np.zeros(4)

        for action in range(4):
            nextState, r, done = env.move(action)

            if nextState != S:
                Q[action] = V[nextState] + r if not done else r
            else:
                Q[action] = abs(np.amin(V)) * -2

            env.set_current_state(S)

        action = np.argmax(Q)

        pos = env.current_position

        if env.grid[pos[0]][pos[1]] != 'G':
            results[pos[0]][pos[1]] = actions[action]

    return results

def runToFinish(envName, V):
    env = Environment(envName)

    currentState = env.reset()

    actions = ["Up", "Left", "Down", "Right"]

    result = []

    done = False
    while not done:
        Q = np.zeros(4)

        for action in range(4):
            nextState, r, done = env.move(action)

            if nextState != currentState:
                Q[action] = V[nextState] + r if not done else r
            else:
                Q[action] = abs(np.amin(V)) * -2

            env.set_current_state(currentState)

        OPT_action = np.argmax(Q)

        currentState, _, done = env.move(OPT_action)
        result.append(actions[OPT_action])

    return result

def run(envName, gridName):
    V, realV, v_ = runForV(envName)
    policy = runForPolicy(envName, v_)

    env = Environment(envName)
    print(env.start)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    im0 = ax0.imshow(V, interpolation="None")
    im1 = ax1.imshow(V, interpolation="None")
    im2 = ax2.imshow(V, interpolation="None")

    for i in range(env.grid_size):
        for j in range(env.grid_size):
            text0 = ax0.text(j, i, env.grid[i][j], ha="center", va="center", color="w")
            text1 = ax1.text(j, i, realV[i, j], ha="center", va="center", color="w")
            text2 = ax2.text(j, i, policy[i][j], ha="center", va="center", color="w")

    ax0.set_title("Grids for %s" % (gridName))
    ax1.set_title("Values for %s"%(gridName))
    ax2.set_title("Policy for %s" % (gridName))
    fig.tight_layout()
    plt.show()

    print(runToFinish(envName, v_))

if __name__ == "__main__":
    for i in range(1, 6):
        run("Grid%d.pkl"%(i),"Grid %d"%(i))
