import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from multiprocessing import Pool
import random

POSSIBLE_DIRS = ['left', 'down', 'right', 'up']
WAREHOUSE_SIZE = 10

class State:
    def __init__(self, reward_params):
        self.actions = [('move', dir) for dir in POSSIBLE_DIRS] + [('stack', i) for i in range(5)] + [('setdown', None), ('pickup', None)]
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = (WAREHOUSE_SIZE - 1, WAREHOUSE_SIZE - 1)
        self.reward_params = reward_params
        self.gamma = 0.99
        self.policy = {}
        self.states = []
        self.CalculateAllStates()
        # print("State space:", len(self.states))     
        
    def CalculateAllStates(self):
        """ Calculate all possible states (discluding impossible ones) stored in self.states """
        skipped = 0
        self.totalSkipped = []
        self.indexValues = [WAREHOUSE_SIZE * 4**5, 4**5, 4**4, 4**3, 4**2, 4**1, 1]
        
        for x in range(WAREHOUSE_SIZE):
            for y in range(WAREHOUSE_SIZE):
                for b1 in range(4):
                    for b2 in range(4):
                        for b3 in range(4):
                            for b4 in range(4):
                                for b5 in range(4):
                                    self.totalSkipped.append(skipped)
                                    # Skip adding state if multiple boxes are marked as being carried
                                    if [b1, b2, b3, b4, b5].count(3) > 1:
                                        skipped += 1
                                        continue
                                        
                                    # Sets initial BoxID of position, based on box initial positions
                                    if (x, y) not in self.box_initial_locations:
                                        self.states.append((x, y, b1, b2, b3, b4, b5, 0))
                                    else:
                                        self.states.append((x, y, b1, b2, b3, b4, b5, self.box_initial_locations.index((x, y)) + 1))
                                 
                                        
    def fastIndex(self, state):
        """ Get the index of the provided state in self.states

        Args:
            state (tuple): Current state of the warehouse

        Returns:
            int: Index of the state in self.states
        """
        absIndex = sum(state[i] * self.indexValues[i] for i in range(len(state)-1))
        return absIndex - self.totalSkipped[absIndex]
              
    
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

    
    # def PrintState(self, state):    
    #     """ Print the current state of the agent

    #     Args:
    #         state (tuple): Current state of the warehouse
    #     """
    #     print("Agent Location: ", state[0], state[1])
    #     print("Boxes: ", state[2:7])
    #     print("BoxID in current location: ", state[7])
        
        
    # def PrintWarehouse(self, state):
    #     """ Print the warehouse with the agent and goal location marked with 'A' and 'G' respectively """        
    #     for i in range(WAREHOUSE_SIZE):
    #         for j in range(WAREHOUSE_SIZE):
    #             if (i,j) == (state[0], state[1]):
    #                 print("A", end = " ")
    #             elif (i,j) == self.goal_location:
    #                 print("G", end = " ")
    #             else:
    #                 print(".", end = " ")
    #         print()
    #     print()
    
    
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
                return (xdir, ydir)

            originalDirection = POSSIBLE_DIRS.index(action[1])
            xmov, ymov = getMov(originalDirection)
            if not (0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE):
                return None

            # left
            direction = (originalDirection - 1) % 4
            xmov, ymov = getMov(direction)
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.05))
            else:
                state_list.append((state, 0.05))

            # right
            direction = (originalDirection + 1) % 4 
            xmov, ymov = getMov(direction)
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                new_state = (x + xmov, y + ymov, *state[2:])
                state_list.append((update_box_id(new_state), 0.05))
            else:
                state_list.append((state, 0.05))

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
            elif state[int(action[1])+2] == 1 and 3 not in state[2:7]: # only let try stack if box is on floor and not carrying a box
                new_state = list(state)
                if self.CheckStackOrder(state, int(action[1])): # stack if correct order
                    new_state[int(action[1])+2] = 2
                else: # unstack
                    for i in range(5):
                        if state[i + 2] == 2:
                            new_state[i + 2] = 1
                state_list.append((tuple(new_state), 1))
            else:
                return None
                
        elif action[0] == "setdown":
            if (state[0], state[1]) != self.goal_location or 3 not in state[2:7]:
                return None
            
            new_state = list(state)
            box_idx = state[2:7].index(3) + 2 
            new_state[box_idx] = 1
            state_list.append((tuple(new_state), 1))
        
        elif action[0] == "pickup":
            # no box initially starts here, the box that's supposed to be here has been picked up, or we are already carrying a box# double check
            if state[7] == 0 or state[state[7]+1] != 0 or 3 in state[2:7]: # note: state[7] starts at 1 so only add 1 to get box index as state idx starts at 0
                return None
            
            new_state = list(state)
            new_state[state[7]+1] = 3
            state_list.append((tuple(new_state), 1))

        else:
            raise Exception("Invalid action")

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
            return self.reward_params['end']

        elif action[0] == 'stack':
            if self.CheckStackOrder(state, int(action[1])):
                return self.reward_params['good_stack']
            else:
                return self.reward_params['bad_stack']
        
        elif action[0] == 'setdown':
            if (state[0], state[1]) == self.goal_location and 3 in state[2:7]:
                return self.reward_params['setdown']

        elif (state[0], state[1]) == self.goal_location: # also based on action?
            return self.reward_params['move_into_goal']

        elif action[0] == 'move':
            return self.reward_params['move']
            
        elif action[0] == 'pickup':
            return self.reward_params['pickup']
        
        else:
            raise Exception("Invalid action")
                
    def ValueIteration(self):
        """ Runs value iteration """
        self.V = np.zeros(len(self.states), dtype=np.float16)
        max_trials = 100
        epsilon = 5.0
        initialize_bestQ = -10000
        curr_iter = 0
        bestAction = np.full(len(self.states), -1, dtype=np.byte)
        
        start = time.time()

        self.P = np.zeros((len(self.actions), len(self.states)), dtype=object)
        
        # Lists to store values for plotting and saving
        residuals = []
        runtimes = []
        iteration_data = []
        
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
            
            residuals.append(max_residual)
            runtime = time.time() - iter_start
            runtimes.append(runtime)
            
            # Store iteration data
            iteration_data.append([curr_iter, max_residual, runtime])
            
            print('Max Residual:', max_residual, "time(m):", runtime / 60)

            if max_residual < epsilon:
                break

        self.policy = bestAction

        end = time.time()
        print('Time taken to converge(m):', (end - start) / 60)
        
        # Plotting the results
        self.plot_results(residuals, runtimes)
        
        # Save results to CSV
        self.save_results_to_csv(iteration_data)
        
    def plot_results(self, residuals, runtimes):
        """ Plot the residuals and runtimes over iterations

        Args:
            residuals (list): List of residuals over iterations
            runtimes (list): List of runtimes over iterations
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(residuals)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Residual')
        ax1.set_title('Residuals over Iterations')

        ax2.plot(runtimes, color='red')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Runtime (s)')
        ax2.set_title('Runtimes over Iterations')

        plt.tight_layout()
        # plt.show()
        
    def save_results_to_csv(self, iteration_data):
        """ Save the iteration data to a CSV file

        Args:
            iteration_data (list): List of iteration data
        """
        df = pd.DataFrame(iteration_data, columns=['Iteration', 'Max Residual', 'Runtime'])
        df.to_csv('value_iteration_results.csv', index=False)
        
    # __cache takes advantage of default parameters to store a local dict that persists between function calls
    def qValue(self, s, action, possible_states, __cache={}): 
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
                indices = np.array([self.fastIndex(state) for state in states],dtype=int)
                probabilities = np.array(probabilities,dtype=np.float16)
                __cache[(s,action)] = (indices,probabilities)
            else:
                indices,probabilities = __cache[(s,action)]
            
            succ_sum = np.sum(probabilities * self.V[indices])
            
            return self.Reward(s, action) + self.gamma * succ_sum
        
        else: # If no possible states, return the initialized bestQ
            return initialize_bestQ

def evaluate_rewards(reward_params):
    move_into_goal, good_stack, bad_stack, setdown, pickup = reward_params
    rewards = {
        'move': -1,
        'move_into_goal': move_into_goal,
        'good_stack': good_stack,
        'bad_stack': bad_stack,
        'setdown': setdown,
        'pickup': pickup,
        'end': 100
    }

    warehouse = State(rewards)
    warehouse.ValueIteration()

    # Get the final residual from the value iteration
    final_residual = warehouse.residuals[-1]

    return rewards, final_residual

def main():
    # Define the range of values you want to explore for each reward parameter
    move_into_goal_range = range(1, 6)
    good_stack_range = range(1, 6)
    bad_stack_range = range(-50, 0, 10) 
    setdown_range = range(1, 6)
    pickup_range = range(1, 6)

    # Create all possible combinations of reward parameters using itertools.product()
    reward_combinations = list(itertools.product(move_into_goal_range, good_stack_range, bad_stack_range, setdown_range, pickup_range))

    # Shuffle the list to randomize the order of exploration
    random.shuffle(reward_combinations)

    # Create a pool of worker processes
    with Pool() as pool:
        # Evaluate reward parameters in parallel
        results = pool.map(evaluate_rewards, reward_combinations)

    best_reward_params = None
    best_residual = float('inf')

    # Find the best reward parameters
    for rewards, final_residual in results:
        if final_residual < best_residual:
            best_reward_params = rewards
            best_residual = final_residual

    # Print the best reward parameters and the corresponding best final residual found
    print("Best reward parameters:", best_reward_params)
    print("Best final residual:", best_residual)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()