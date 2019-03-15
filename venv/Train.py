from Environment import Environment
from Agent import Agent
import numpy as np
from collections import deque

if __name__ == '__main__':
    env = Environment('Grid1.pkl')

    episodeCounter = 0
    totalIter = 0
    iter = 0

    agent = Agent(env)

    print(env.grid)
    print(env.start)

    errors = deque(maxlen=agent.stateSize)

    while (True):
        error, done = agent.explore()

        iter += 1
        totalIter += 1
        errors.append(error)

        if agent.isAllNodesAreVisited() and len(errors) > agent.stateSize / 2 and np.mean(errors) < 1.0:
            print("Completed at %d iteration" % (totalIter))
            break

        if episodeCounter > 1000:
            print("Break Out!")
            break

        if done:
            if agent.isAllNodesAreVisited() and np.mean(errors) < 1.0 and episodeCounter > 1:
                print("Done and Completed at %d iteration" % (totalIter))
                break

            print("%d. episode is done at %d"%(episodeCounter, iter))
            iter = 0
            episodeCounter += 1
            env.reset()
            agent.updateCurrentState(env.current_position)

    print(agent.V)
    print(agent.V_visited)
    print(agent.R)
    
    env.reset()
    agent.updateCurrentState(env.current_position)
    print("------------------------------")
    print("Optimal Policy:")
    done = False
    while not done:
        agent.updateCurrentState(env.current_position)
        action = agent.act(env.current_position)
        _, r, done = env.move(action)
        print(action, r, agent.currentPosition)