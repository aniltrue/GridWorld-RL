from Environment import Environment

class Agent:
    def __init__(self, env: Environment, discountRate = 0.995, actionSize = 4):
        '''
        Initiate the Agent
        :param env: The Environment where the Agent plays.
        :param discountRate: Discount rate of cumulative rewards. Must be between 0.0 and 1.0
        :param actionSize: Possible actions
        '''
        self.env = env
        self.stateSize = env.grid_size * env.grid_size

        assert actionSize > 0, "actionSize must be positive"
        self.actionSize = actionSize

        assert discountRate >= 0.0 and discountRate <= 1.0, "discountRate must be in range [0.0, 1.0]"
        self.discountRate = discountRate

    def train(self, TDLimit = 0.1, printDetails = True, **kwargs):
        '''
        Implement this method, Not Call!
        :param TDLimit: TDLimit: If TDError is minumum, Agent is converged.
        :param printDetails: Printing the details while training.
        :param kwargs: Some methods needs some more arguments.
        :return: None
        '''

        assert TDLimit >= 0.0, "TDLimit must be 0 or positive"

    def act(self, state: int, isTraining = False) -> int:
        '''
        Implement this method, Not Call!
        Do Not Call Act() Before Training!
        :param state: Current State as Integer not Position
        :param isTraining: Some Agents act different while training.
        :return: Action the Agent decided.
        '''

        assert state >= 0 and state < self.stateSize, "state is out of bonds! [0, %d]"%(self.stateSize)