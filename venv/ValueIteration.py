# Anıl Doğru S007130 Department of Computer Science

from Environment import Environment
from Agent import Agent
import pickle as pkl
import numpy as np

class ValueIterAgent(Agent):
    def __init__(self, env: Environment, discountRate = 0.995, actionSize = 4):
        '''
        Initiate the Agent
        :param env: The Environment where the Agent plays.
        :param discountRate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param actionSize: Possible actions
        '''
        super().__init__(env, actionSize = actionSize, discountRate=discountRate)

        self.V = np.zeros(self.stateSize)
        self.OPT_Policy = np.zeros(self.stateSize, dtype=np.int)

    def train(self, TDLimit = 0.0, printDetails = False, **kwargs):
        '''
        Training Agent to fill the Value Table.
        :param TDLimit: If TDError is minumum, Value Table is converged.
        :param printDetails: Printing details of all episodes
        :param kwargs: This is not effective.
        :return: None
        '''
        super().train(TDLimit, printDetails, kwargs)
        error = 1.0
        iter = 0

        while error > TDLimit:
            error = 0.0

            for (currentState, currentV) in enumerate(self.V):
                self.env.set_current_state(currentState)
                pos = self.env.current_position

                if self.env.grid[pos[0]][pos[1]] == 'G':
                    continue

                Q = np.zeros(self.actionSize, dtype=np.int)

                for action in range(self.actionSize):
                    self.env.set_current_state(currentState)

                    nextState, reward, done = self.env.move(action)

                    Q[action] = reward + self.discountRate * self.V[nextState] if not done else reward

                V = np.amax(Q)
                optAction = np.argmax(Q)

                error = max(error, abs(currentV - V))

                self.V[currentState] = V
                self.OPT_Policy[currentState] = optAction

            if printDetails:
                print("%d. iteration is completed. Max. error: %d"%(iter, error))
            iter += 1
    
    def act(self, state: int, isTraining = False) -> int:
        '''
        While traning, Optimal Policy is also generated. With Greedy Approach, Optimal Policy is generated.
        DO NOT CALL THIS METHOD WITHOUT TRAINING! 
        :param state: Current State as Integer not Position
        :param isTraining: It is not effective.
        :return: Action as integer
        '''
        super().act(state, isTraining)
        return self.OPT_Policy[state]
        
if __name__ == "__main__":
    solutions = {}

    for i in range(1, 6):
        envName = "Grid%d.pkl"%(i)
        print("-------------------------%s-------------------------"%(envName))

        env = Environment(envName)

        agent = ValueIterAgent(env)
        agent.train()

        solution = np.reshape(agent.V, (env.grid_size, env.grid_size))
        solutions[envName] = solution

    solFile = open("Solutions.pkl", "wb")
    pkl.dump(solutions, solFile)
    solFile.close()