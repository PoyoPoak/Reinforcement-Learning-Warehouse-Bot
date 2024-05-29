import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Example of a state: (agent_x, agent_y, b1_status, b2_status, b3_status, b4_status, b5_status, BoxID of box's initial location)

POSSIBLE_DIRS = ['left', 'down', 'right', 'up']
WAREHOUSE_SIZE = 10

class State:
    def __init__(self):
        self.actions = [('move', dir) for dir in POSSIBLE_DIRS] + [('stack', i) for i in range(5)] + [('setdown', None), ('pickup', None)]
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = (WAREHOUSE_SIZE - 1, WAREHOUSE_SIZE - 1)
        self.gamma = 0.99
        self.policy = {}
        self.states = []
        self.CalculateAllStates()
        print("State space:",len(self.states))     
        
    def CalculateAllStates(self):
        """ Calculate all possible states (discluding impossible ones) stored in self.states """
        for x in range(WAREHOUSE_SIZE):
            for y in range(WAREHOUSE_SIZE):
                for b1 in range(4):
                    for b2 in range(4):
                        for b3 in range(4):
                            for b4 in range(4):
                                for b5 in range(4):
                                    # Skip adding state if multiple boxes are marked as being carried
                                    if [b1, b2, b3, b4, b5].count(3) > 1:
                                        continue
                                        
                                    # Sets initial BoxID of position, based on box initial positions
                                    if (x,y) not in self.box_initial_locations:
                                        self.states.append((x, y, b1, b2, b3, b4, b5, 0))
                                    else:
                                        self.states.append((x, y, b1, b2, b3, b4, b5, self.box_initial_locations.index((x,y)) + 1))
              
    
    def CheckGoalState(self, state):
        """ Check if the current state is the goal state

        Args:
            state (tuple): Current state of the warehouse

        Returns:
            bool: True if the state is the goal state, False otherwise.
        """
        return state == (9, 9, 2, 2, 2, 2, 2, 0)       
                                    
    
    def CheckStackOrder(self, state, box):
        """ Check if the box can be stacked on top of the current stack

        Args:
            state (tuple): Current state of the warehouse
            box (int): BoxID of the box to be stacked (0-4)

        Returns:
            bool: True if the box can be stacked, False otherwise.
        """
        # Check if the box is already stacked
        if state[box + 2] == 2:  
            return False
        
        current_stack = [i for i in range(5) if state[i + 2] == 2]
        
        # No boxes stacked, any box can be stacked
        if not current_stack:  
            return True
        return all(box < stacked_box for stacked_box in current_stack)

    
    def PrintState(self, state):    
        """ Print the current state of the agent

        Args:
            state (tuple): Current state of the warehouse
        """
        print("Agent Location: ", state[0], state[1])
        print("Boxes: ", state[2:7])
        print("BoxID in current location: ", state[7])
        
        
    def PrintWarehouse(self, state):
        """ Print the warehouse with the agent and goal location marked with 'A' and 'G' respectively """        
        for i in range(WAREHOUSE_SIZE):
            for j in range(WAREHOUSE_SIZE):
                if (i,j) == (state[0], state[1]):
                    print("A", end = " ")
                elif (i,j) == self.goal_location:
                    print("G", end = " ")
                else:
                    print(".", end = " ")
            print()
        print()
    
    
    def Transition(self, state, action):
        """ Our transition function, returns a list of possible states and their probabilities.

        Args:
            state (tuple): Current state of the warehouse
            action (tuple): Action to be taken (e.g. ('move', 'left') or ('stack', 2)

        Returns:
            list: List of possible states and their probabilities. 
                    Ex: [((1, 2, 0, 0, 0, 0, 0, 0), 0.8), ((1, 3, 0, 0, 0, 0, 0, 0), 0.2) ...]
        """
        state_list = []
        
        if action[0] == 'move':
            x = state[0]
            y = state[1]

            def update_box_id(new_state):  
                x, y = new_state[0], new_state[1]
                if (x, y) in self.box_initial_locations:
                    box_id = self.box_initial_locations.index((x, y)) + 1
                else:
                    box_id = 0
                return new_state[:7] + (box_id,)

            def getMov(dir):
                xdir = [0, 1, 0, -1][dir]
                ydir = [-1, 0, 1, 0][dir]
                return (xdir,ydir)

            originalDirection = POSSIBLE_DIRS.index(action[1])
            xmov,ymov = getMov(originalDirection)
            if not(0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE):
                return None

            # left
            direction = (originalDirection - 1) % 4
            xmov,ymov = getMov(direction)
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.05))
            else:
                state_list.append((state,0.05))

            # right
            direction = (originalDirection + 1) % 4 
            xmov,ymov = getMov(direction)
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.05))
            else:
                state_list.append((state,0.05))

            # double & regular move
            xmov, ymov = getMov(originalDirection)
            xmov *= 2
            ymov *= 2
            
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.1))
                xmov, ymov = getMov(originalDirection)
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.8))  
            else:
                xmov, ymov = getMov(originalDirection)
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.9))   
           
        elif action[0] == "stack":
            if (state[0], state[1]) != self.goal_location:
                return None
            else:
                new_state = list(state)
                if self.CheckStackOrder(state, int(action[1])): # stack
                    new_state[int(action[1])+2] = 2 
                else: # unstack
                    for i in range(5):
                        if state[i + 2] == 2:
                            new_state[i + 2] = 1
                state_list.append((tuple(new_state), 1))
                
        elif action[0] == "setdown":
            if (state[0], state[1]) != self.goal_location or 3 not in state[2:7]:
                return None
            
            new_state = list(state)
            box_idx = state[2:7].index(3) + 2 
            new_state[box_idx] = 1
            state_list.append((tuple(new_state), 1))
        
        elif action[0] == "pickup":
            # no box initially starts here, the box that's supposed to be here has been picked up, or we are already carrying a box# double check
            if state[7] == 0 or state[state[7]+2] != 0 or 3 in state[2:7]:
                return None
            
            new_state = list(state)
            new_state[state[7]+1] = 3
            state_list.append((tuple(new_state), 1))

        else:
            raise Exception("Invalid action")

        # Test prints
        # self.PrintState(state)
        # self.PrintWarehouse(state)
        # print("--------------------------------------")
        # for s in state_list:
        #     self.PrintState(s[0])
        #     self.PrintWarehouse(s[0])

        return state_list

    def Reward(self, state, action):
        """ Reward function. Returns the reward for a given state and action.

        Args:
            state (tuple): Current state of the warehouse
            action (tuple): Action to be taken (e.g. ('move', 'left') or ('stack', 2)

        Returns:
            float: Reward for the given state and action
        """
        if action[0] == 'end':
            return 100

        elif action[0] == 'stack':
            if self.CheckStackOrder(state, int(action[1])):
                return 10
            else:
                return -50 
        
        elif action[0] == 'setdown':
            if (state[0], state[1]) == self.goal_location and 3 in state[2:7]:
                return 5 

        # elif (state[0], state[1]) == self.goal_location: # also based on action?
        #     return 2

        elif action[0] == 'move':
            return -1
            
        elif action[0] == 'pickup':
            return 5 
        
        else:
            raise Exception("Invalid action")
                
    def ValueIteration(self):
        """ Runs value iteration """
        self.V = np.zeros(len(self.states),dtype=np.float16)
        max_trials = 100
        epsilon = 0.01
        initialize_bestQ = -10000
        curr_iter = 0
        bestAction = np.full(len(self.states), -1,dtype=np.byte)
        
        start = time.time()

        self.P = np.zeros((len(self.actions),len(self.states)),dtype=object)
        
        while curr_iter < max_trials:
            iter_start = time.time()
            max_residual = 0
            curr_iter += 1
            print('Iteration: ', curr_iter)
            
            bestQ = np.full_like(self.V, initialize_bestQ,dtype=np.float16)
            # Loop over states to calculate values
            for idx, s, in enumerate(self.states):
                if self.CheckGoalState(s): # Check for goal state
                    bestAction[idx] = 0
                    bestQ[idx] = self.Reward(s, ('end', None))  
                    continue
                
                for na, action in enumerate(self.actions):
                    if self.P[na,idx] == None: # If no possible states, continue
                        continue

                    # If this state action pair hasn't been evaluated yet, store it in probabilities
                    elif self.P[na,idx] == 0:
                        self.P[na,idx] = self.Transition(s, action)

                    qaction = self.qValue(s, action, self.P[na,idx])

                    if qaction > bestQ[idx]:
                        bestQ[idx] = qaction
                        bestAction[idx] = na
            
            residual = np.abs(bestQ - self.V)
            self.V = bestQ
            max_residual = max(max_residual,np.max(residual))

            print('Max Residual:', max_residual, "time:",(time.time() - iter_start) / 60)

            if max_residual < epsilon:
                break

        self.policy = bestAction

        end = time.time()
        print('Time taken to solve (minutes): ', (end - start) / 60)
        
    # __cache takes advantage of default paramaters to store a local dict that persists between function calls
    def qValue(self, s, action, possible_states, __cache = {}): 
        """ Calculate the Q-value of a given state and action

        Args:
            s (tuple): Current state of the warehouse   
            action (tuple): Action to be taken (e.g. ('move', 'left') or ('stack', 2)
            possible_states (list): List of next possible states with action probabilities

        Returns:
            float: Q-value of the given state and action
        """
        initialize_bestQ = -10000
        
        if possible_states is not None: # If there are possible states

            if (s,action) not in __cache:
                states,probabilities = zip(*possible_states)
                indices = np.array([self.states.index(state) for state in states],dtype=int)
                probabilities = np.array(probabilities,dtype=np.float16)
                __cache[(s,action)] = (indices,probabilities)
            else:
                indices,probabilities = __cache[(s,action)]
            
            succ_sum = np.sum(probabilities * self.V[indices])
            
            return self.Reward(s, action) + self.gamma * succ_sum
        
        else: # If no possible states, return the initialized bestQ
            return initialize_bestQ
        
    def EveryVisitMonteCarlo(self, num_episodes = 500, gamma = 0.99, max_episode_length = 10000):
        """ Runs the every visit Monte Carlo algorithm """
        self.V = {state: np.random.rand() for state in self.states} # Randomly initialize value function
        blankDict = {state: [] for state in self.states}
        values_of_inital_state = []
        for i in tqdm(range(num_episodes)):
            state = (0,0,0,0,0,0,0,0) # Initialize episode
            action = max([(self.testQ(state, action), action) for action in self.getPossibleActions(state)], key=lambda x: x[0])[1] # Choose action with highest value
            episode = []
            reward = self.Reward(state, action)
            episode.append((state, action, reward))
            episode_length = 0
            while not self.CheckGoalState(state) and episode_length != max_episode_length: # Generate episode
                # self.PrintWarehouse(state)
                episode_length += 1
                states = [state for state, probability in self.Transition(state, action)]
                probabilities = [probability for state, probability in self.Transition(state, action)]
                state = random.choices(states, probabilities, k=1)[0] # Randomly choose next state
                action = max([(self.testQ(state, action), action) for action in self.getPossibleActions(state)], key=lambda x: x[0])[1] # Choose action with highest value
                reward = self.Reward(state, action)
                episode.append((state, action, reward))

            episode_values = blankDict.copy() # Set up new value calc and storage
            value = 0
            episode.reverse()
            for (state, action, reward) in episode: # Calculate and store discounted cumulative reward
                value = reward + gamma * value
                episode_values[state].append(value)
            for state in episode_values.keys(): # Update value function with mean of episode values
                if episode_values[state] != []:
                    self.V[state] = np.mean(episode_values[state])
            
            values_of_inital_state.append(self.V[(0,0,0,0,0,0,0,0)]) # Store value of initial state
        # Plot the value function of the initial state
        self.plot(range(1, num_episodes+1), values_of_inital_state, "Episodes", "Average Value", "Average Value of Initial State vs Episodes")
        
        # display path
        state = (0,0,0,0,0,0,0,0) # Initialize episode
        action = max([(self.testQ(state, action), action) for action in self.getPossibleActions(state)], key=lambda x: x[0])[1] # Choose action with highest value
        episode_length = 0
        self.PrintWarehouse(state)
        while not self.CheckGoalState(state) and episode_length != 30: # Generate episode
            episode_length += 1
            states = [state for state, probability in self.Transition(state, action)]
            probabilities = [probability for state, probability in self.Transition(state, action)]
            state = random.choices(states, probabilities, k=1)[0] # Randomly choose next state
            action = max([(self.testQ(state, action), action) for action in self.getPossibleActions(state)], key=lambda x: x[0])[1] # Choose action with highest value
            self.PrintWarehouse(state)



    def getPossibleActions(self, state):
        """ Get the possible actions from a state """
        return [action for action in self.actions if self.Transition(state, action) is not None]
    
    def testQ(self, state, action):
        """ Calculate the Q-value of a given state and action given the value of the immediate next states """
        value = 0
        for state, probability in self.Transition(state, action):
            value += probability * self.V[state]
        return value
    
    def plot(self, x, y, x_label, y_label, title):
        plt.plot(x, y, 'o', label='original data')

        # Calc regression line
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        regression_line = polynomial(x)
        plt.plot(x, regression_line, 'r--', label='regression line')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()

warehouse = State()
warehouse.EveryVisitMonteCarlo()
# print()
# test()
