from Environment import Environment
from Agent import Agent
import numpy as np
import random as rnd

class SarsaAgent(Agent):
    def __init__(self, env: Environment, discountRate = 0.995, epsilon = 1.0, epsilonDecay = 0.995, epsilonMin = 0.1, alpha = 0.2):
        '''
        Initiate the Agent with hyperparameters
        :param env: The Environment where the Agent plays.
        :param discountRate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param epsilon: Starting epsilon for e-greedy
        :param epsilonDecay: epsilon *= epsilon * epsilonDecay after all e-greedy. Less than 1.0
        :param epsilonMin: Minumum epsilon to avoid overestimation. Must be positive or zero
        :param alpha: To update Q values softly. 0 < alpha <= 1.0
        '''
        super().__init__(env, discountRate)

        assert epsilon >= 0.0, "epsilon must be 0 or positive"
        self.epsilon = epsilon

        assert epsilonDecay >= 0.0 and epsilonDecay <= 1.0, "epsilonDecay must be in range [0.0, 1.0]"
        self.epsilonDecay = epsilonDecay

        assert epsilonMin >= 0.0, "epsilonMin must be 0 or positive"
        self.epsilonMin = epsilonMin

        assert alpha > 0.0 and alpha <= 1.0, "alpha must be in range (0.0, 1.0]"
        self.alpha = alpha

        self.Q = np.zeros((self.stateSize, self.actionSize))

    def train(self, TDLimit = 0.01, printDetails = True, **kwargs):
        '''
        Training the Agent to fill Q Table
        :param TDLimit: If TDError is minumum, Q Table is converged.
        :param printDetails: Printing details of all episodes
        :param kwargs: "maxEpisode" is the limit agent plays max. episode. Default: 100
        :return: None
        '''
        super().train(TDLimit, printDetails)

        maxEpisode = 1000

        if len(kwargs) == 1:
            maxEpisode = kwargs["maxEpisode"]
            assert maxEpisode > 0, "maxEpisode must be positive"

        for episode in range(maxEpisode):
            currentState = self.env.reset()
            action = self.act(currentState, True)

            done = False
            maxTD = 0.0

            while not done:
                nextState, reward, done = env.move(action)

                nextAction = self.act(nextState, True)

                q = reward + self.Q[nextState][nextAction] if not done else reward
                TD = q - self.Q[currentState, action]

                self.Q[currentState][action] += self.alpha * TD

                maxTD = max(maxTD, abs(TD))
                currentState = nextState
                action = nextAction

            if printDetails:
                print("%d. episode completed. Max. TD: %f" % (episode, maxTD))

            if maxTD < TDLimit:
                if printDetails:
                    print("Converged!")

                break


    def act(self, state: int, training = False) -> int:
        '''
        DO NOT CALL THIS METHOD WITHOUT TRAINING!
        :param state: Current State as Integer not Position
        :param training: If training use e-greedy, otherwise greedy to choise action.
        :return: Action as integer
        '''
        super().act(state, training)

        if training and rnd.random() <= self.epsilon:
            self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)
            return rnd.randrange(self.actionSize)

        return np.argmax(self.Q[state])

if __name__ == "__main__":
    env = Environment("Grid1.pkl")

    agent = SarsaAgent(env)

    agent.train()