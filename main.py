import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import pickle
# Example of a state: (agent_x, agent_y, b1_status, b2_status, b3_status, b4_status, b5_status, BoxID of box's initial location)
# Box Status: 0: In starting space, 1: Sitting in the goal space, 2: Stacked in the goal, 3: currently being carried

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
        
        # Warehouse layout visualized
        # ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** B2 **
        # ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** ** B1 ** ** ** **
        # ** ** ** ** ** ** ** ** ** **
        # ** ** ** ** B3 ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** **
        # ** ** B5 ** ** ** ** ** ** **
        # ** ** ** ** ** ** ** ** ** **
        # ** B4 ** ** ** ** ** ** ** G
        
    def PrintWarehouse(self, state, action=None, action_num=None):
        """ Print the warehouse with the agent and goal location marked with 'A' and 'G' respectively """        
        for i in range(WAREHOUSE_SIZE):
            for j in range(WAREHOUSE_SIZE):
                if (i,j) == (state[0], state[1]):
                    print("A", end = " ")
                elif (i,j) in self.box_initial_locations:
                    print(f"B", end = " ")
                elif (i,j) == self.goal_location:
                    print("G", end = " ")
                else:
                    print(".", end = " ")
            print()
        print("- B1:", state[2], "- B2:", state[3], "- B3:", state[4], "- B4:", state[5], "- B5:", state[6], "action:", action, action_num)
        print("reward:", self.Reward(state, action), "\n")
    
    
    def Transition(self, state, action, __cache = {}):
        """ Our transition function, returns a list of possible states and their probabilities.

        Args:
            state (tuple): Current state of the warehouse
            action (tuple): Action to be taken (e.g. ('move', 'left') or ('stack', 2)

        Returns:
            list: List of possible states and their probabilities. 
                    Ex: [((1, 2, 0, 0, 0, 0, 0, 0), 0.8), ((1, 3, 0, 0, 0, 0, 0, 0), 0.2) ...]
        """

        if (state, action) in __cache:
            return __cache[(state, action)]

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
            return 100

        elif action[0] == 'stack':
            if self.CheckStackOrder(state, int(action[1])):
                return 10
            else:
                return -50 
        
        elif action[0] == 'setdown':
            if (state[0], state[1]) == self.goal_location and 3 in state[2:7]:
                return 5 

        elif action[0] == 'move':
            return -0.05
            
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
        
    def EveryVisitMonteCarlo(self, num_episodes = 100000, max_episode_length = 300, epsilon = 0.2, gamma = 0.9):
        print("Number of episodes:", num_episodes, "Max episode length:", max_episode_length, "Epsilon:", epsilon, "Gamma:", gamma)
        """ Runs the every visit Monte Carlo algorithm """
        cwd = os.path.dirname(os.path.abspath(__file__))
        func_file_path = os.path.join(cwd, 'valueFunctions\\policy.pkl')
        self.policy = {}
        self.Q = np.random.rand(len(self.states) * len(self.actions)).astype(np.float16)
        state_action_combinations = [(state, action) for state in self.states for action in self.actions]
        self.Q_to_idx = {state_action: idx for idx, state_action in enumerate(state_action_combinations)} # Map state-action pairs to Q-value index
        self.update_policy(self.states, epsilon)

        episode_times = [] # Data to store
        episode_stacks = []
        episode_lengths = []
        accumulated_rewards = []
        setdowns = 0
        pickups = 0
        for i in tqdm(range(num_episodes)):
            episode_start_time = time.time()
            episode_stacks.append(0)
            accumulated_reward = 0

            episode_length = 0
            episode = [] # Initialize episode
            state = (0,0,0,0,0,0,0,0)
            while not self.CheckGoalState(state) and episode_length != max_episode_length: # Generate episode
                action = self.policy[state]
                reward = self.Reward(state, action)
                episode.append((state, action, reward))
                states, probabilities = zip(*self.Transition(state, action)) # (state, prob) -> [states], [probs]
                state = random.choices(states, probabilities, k=1)[0] # Randomly choose next state

                episode_length += 1
                accumulated_reward += reward # Collect information
                if action[0] == 'stack':
                    episode_stacks[-1] += 1
                if action[0] == 'setdown':
                    setdowns += 1
                if action[0] == 'pickup':
                    pickups += 1
            episode_lengths.append(episode_length)
            accumulated_rewards.append(accumulated_reward)

            state_action_rewards_lists = {}
            encountered_state_actions = set()
            encountered_states = set()
            cumulative_reward = 0
            # Calculate Q-Values
            for (state, action, reward) in reversed(episode):
                cumulative_reward = reward + gamma * cumulative_reward

                if (state, action) not in encountered_state_actions: # Create list of each state-action encounter in episode
                    state_action_rewards_lists[(state, action)] = np.array([], dtype=np.float16) # Initialize list
                state_action_rewards_lists[(state, action)] = np.append(state_action_rewards_lists[(state, action)], cumulative_reward) # Append reward to list

                encountered_state_actions.add((state, action)) # Add state-action to encountered for averaging
                encountered_states.add(state) # Add state to encountered for policy update
                
            # self.print_policy(f"policy{i}.txt", episode)

            for state_action in encountered_state_actions: # Update Q-values
                self.Q[self.Q_to_idx[state_action]] = np.mean(state_action_rewards_lists[state_action])
            
            self.update_policy(encountered_states, epsilon)
            
            # if epsilon > 0.05:
            #     epsilon = epsilon * 0.9999

            episode_times.append(time.time() - episode_start_time) # Store episode time
        
        with open(func_file_path, 'wb') as f:
            pickle.dump(self.policy, f)
        
        print("setdowns:", setdowns, "ratio to episodes:", setdowns/num_episodes)
        print("pickups:", pickups, "ratio to episodes:", pickups/num_episodes)
        print("Average episode time:", np.mean(episode_times), "seconds, Std:", np.std(episode_times), ", Total time:", np.sum(episode_times)/60, "minutes")
        # self.plot(range(1, num_episodes+1), episode_times, "Episodes", "Time (s)", "Episode Time vs Episodes")
        # Plot the total rewards per episode
        self.plot(range(1, num_episodes+1), accumulated_rewards, "Episodes", "Accumulated Reward", "Accumulated Reward vs Episodes")
        # Plot the number of stacks per episode
        self.plot(range(1, num_episodes+1), episode_stacks, "Episodes", "Stacks", "Stacks vs Episodes")
        # Plot the number of stacks per episode (stacks <= 1)
        # stacks_less_or_equal_1 = [s for s in episode_stacks if s <= 1]
        stacks_less_or_equal_1_not_0 = [s for s in episode_stacks if s <= 1 and s != 0]
        print("Number of stacks <= 1 and != 0:", len(stacks_less_or_equal_1_not_0))
        # self.plot(range(1, len(stacks_less_or_equal_1)+1), stacks_less_or_equal_1, "Episodes", "Stacks", "Stacks vs Episodes (Stacks <= 1)")
        # Plot the episode lengths
        # lengths_not_max = [l for l in episode_lengths if l != max_episode_length]
        # print("Number of episodes not reaching max length:", len(lengths_not_max))
        # self.plot(range(1, num_episodes+1), episode_lengths, "Episodes", "Episode Length", "Episode Length vs Episodes")
        
        self.test_every_visit_monte_carlo(max_episode_length)
        

    def update_policy(self, states, epsilon):
        """ Update policy based on epsilon greedy Q-values """
        for state in states:
            possible_actions = [action for action in self.actions if self.Transition(state, action) is not None]

            best_action = max([(self.Q[self.Q_to_idx[(state, action)]], action) for action in possible_actions])[1] # Best Q-val for list (Q-val, action)
            suboptimal_actions = [action for action in possible_actions if action != best_action]

            possible_actions = [best_action] + suboptimal_actions # Put best action first
            action_probabilities = [1 - epsilon + epsilon / len(possible_actions)] + [epsilon / len(possible_actions)] * (len(possible_actions) - 1) # Prob for each action, best first
            self.policy[state] = random.choices(possible_actions, action_probabilities, k=1)[0] # Choose action with epsilon greedy policy
    
    def test_every_visit_monte_carlo(self, max_episode_length, load_value_function = False):
        """ Test the every visit Monte Carlo algorithm """
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_file_path = os.path.join(cwd, 'paths\\agent_path.txt')
        if load_value_function:
            func_file_path = os.path.join(cwd, 'valueFunctions\\policy.pkl')
            with open(func_file_path, 'rb') as f:
                self.policy = pickle.load(f)
        original_max_episode_length = max_episode_length
        max_episode_length *= 2
        original_stdout = sys.stdout
        with open(output_file_path, 'w') as sys.stdout:
            print("max_episode_length:", max_episode_length)
            state = (0,0,0,0,0,0,0,0) # Initialize episode
            episode_length = 0
            while not self.CheckGoalState(state) and episode_length != max_episode_length: # Generate episode
                episode_length += 1
                action = self.policy[state]
                self.PrintWarehouse(state, action, episode_length+1)
                if episode_length == original_max_episode_length:
                    print("---- Training Max episode length reached ----")
                if episode_length == max_episode_length:
                    print("---- Max episode length reached ----")
                states, probabilities = zip(*self.Transition(state, action))
                state = random.choices(states, probabilities, k=1)[0] # Randomly choose next state
        sys.stdout = original_stdout

    def plot(self, x, y, x_label, y_label, title):
        plt.plot(x, y, 'o', label='original data')
        # plt.bar(np.arange(len(y)),y, label='original data')

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

    def print_policy(self, filename, episode):
        original_stdout = sys.stdout
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warehouse_policies\\" + filename)
        with open(output_file_path, 'w') as sys.stdout:
            for y in range(WAREHOUSE_SIZE):
                for x in range(WAREHOUSE_SIZE):
                    spacing = "\t\t\t"
                    if (x,y) in self.box_initial_locations:
                        box = self.box_initial_locations.index((x, y)) + 1
                        print(f"B{box}{spacing}", end = "")
                    elif (x,y) == self.goal_location:
                        print(f"G{spacing}", end = "")
                    else:
                        print(f".{spacing}", end = "")
                print()
            print("\n")

            # for y in range(WAREHOUSE_SIZE):
            #     for x in range(WAREHOUSE_SIZE):
            #         box_in_loc = self.box_initial_locations.index((x, y)) + 1 if (x, y) in self.box_initial_locations else 0
            #         actions = [action for action in self.actions if self.Transition((x, y, 0, 0, 0, 0, 0, box_in_loc), action) is not None]
            #         for action in actions:
            #             if action == self.policy[(x, y, 0, 0, 0, 0, 0, box_in_loc)]:
            #                 print("action", end = "-")
            #             print(f"({x}, {y}) {action} {self.Q[self.Q_to_idx[((x, y, 0, 0, 0, 0, 0, box_in_loc), action)]]:.2f}", end = " ")
            #         print()

            for (state, agent_action, reward) in episode:
                # for y in range(WAREHOUSE_SIZE):
                #     for x in range(WAREHOUSE_SIZE):
                        # box_in_loc = self.box_initial_locations.index((x, y)) + 1 if (x, y) in self.box_initial_locations else 0
                actions = [action for action in self.actions if self.Transition(state, action) is not None]
                space = "\t| "
                if agent_action[1] == 'up' or agent_action[1] == 'left' or agent_action[1] == 'down':
                    space = "\t\t| "
                print(f"({state[0]}, {state[1]}) Agent action: {agent_action} Reward: {reward}", end = space)
                for action in actions:
                    print(f"({action} {self.Q[self.Q_to_idx[(state, action)]]:.2f})", end = " ")
                print()
            print()

            for (state, agent_action, reward) in episode:
                for y in range(WAREHOUSE_SIZE):
                    for x in range(WAREHOUSE_SIZE):
                        box = self.box_initial_locations.index((x, y)) + 1 if (x, y) in self.box_initial_locations else 0
                        action = self.policy[(x, y, *state[2:7], box)]
                        if (x, y) == (state[0], state[1]):
                            print(f"A\t\t\t", end = "")
                        elif action[1] == 'up':
                            print(f"{action[0]} {action[1]}\t\t", end="")
                        elif action[0] == 'pickup':
                            print(f"{action[0]} {box}\t", end="")
                        else:
                            print(f"{action[0]} {action[1]}\t", end="")
                    print()
                print(f"action: {agent_action} reward: {reward} b1: {state[2]} b2: {state[3]} b3: {state[4]} b4: {state[5]} b5: {state[6]}\n")
        sys.stdout = original_stdout

warehouse = State()
warehouse.EveryVisitMonteCarlo()
# warehouse.test_every_visit_monte_carlo(500, True)
# print()
# test()
