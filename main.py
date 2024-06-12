import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import sys

POSSIBLE_DIRS = ['left', 'down', 'right', 'up']
WAREHOUSE_SIZE = 10

class State:
    def __init__(self):
        self.actions = [('move', dir) for dir in POSSIBLE_DIRS] + [('stack', i) for i in range(5)] + [('setdown', None), ('pickup', None)]
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = (WAREHOUSE_SIZE - 1, WAREHOUSE_SIZE - 1)
        self.policy = {}
        self.states = []
        self.CalculateAllStates()
        self.residuals = []
        self.iteration_times = []


    def CalculateAllStates(self):
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
        absIndex = sum(state[i] * self.indexValues[i] for i in range(len(state)-1))
        return absIndex - self.totalSkipped[absIndex]
              
              
    def CheckGoalState(self, state):
        return state == (9, 9, 2, 2, 2, 2, 2, 0)       
                                    
    
    def CheckStackOrder(self, state, box):
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
    
    
    def Transition(self, state, action):
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
            return 5 * state[7]
        
        else:
            raise Exception("Invalid action")
                
                
    def ValueIteration(self, gamma):
        self.V = np.zeros(len(self.states), dtype=np.float16)
        max_trials = 1000
        epsilon = 0.00001
        initialize_bestQ = -10000
        curr_iter = 0
        self.P = np.zeros((len(self.actions), len(self.states)), dtype=object)
        self.residuals = []
        self.policy_changes = []
        self.iteration_times = []

        start = time.time()
        while curr_iter < max_trials:
            iter_start = time.time()
            max_residual = 0
            curr_iter += 1
            
            bestQ = np.full_like(self.V, initialize_bestQ, dtype=np.float16)
            new_policy = np.full(len(self.states), -1, dtype=np.byte)
            # Loop over states to calculate values
            for idx, s, in enumerate(self.states):
                if self.CheckGoalState(s): # Check for goal state
                    new_policy[idx] = 0
                    bestQ[idx] = self.Reward(s, ('end', None))  
                    continue
                
                for na, action in enumerate(self.actions):
                    if self.P[na,idx] == None: # If no possible states, continue
                        continue

                    # If this state action pair hasn't been evaluated yet, store it in probabilities
                    elif self.P[na,idx] == 0:
                        self.P[na,idx] = self.Transition(s, action)

                    qaction = self.qValue(s, action, self.P[na,idx], gamma)

                    if qaction > bestQ[idx]:
                        bestQ[idx] = qaction
                        new_policy[idx] = na
            
            residual = np.abs(bestQ - self.V)
            self.V = bestQ
            max_residual = max(max_residual,np.max(residual))

            # Store residuals in the object
            self.residuals.append(max_residual)

            # Calculate policy changes
            if curr_iter > 1:
                policy_change_count = np.sum(new_policy != self.policy)
                self.policy_changes.append(policy_change_count)
            else:
                self.policy_changes.append(0)

            self.policy = new_policy

            runtime = time.time() - iter_start
            self.iteration_times.append(runtime)

            print('Iteration:', curr_iter, 'Max Residual:', max_residual, "time(s):", runtime)

            if max_residual < epsilon:
                break

        end = time.time()
        print('Time taken to solve (minutes): ', (end - start) / 60)
        
        return curr_iter
		
		
    def qValue(self, s, action, possible_states, gamma, __cache = {}): 
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
            
            return self.Reward(s, action) + gamma * succ_sum
        
        else: # If no possible states, return the initialized bestQ
            return initialize_bestQ
        
        
    def VI_plot(self):
        matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Max Residual', color=color)
        ax1.plot(range(len(self.residuals)), self.residuals, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Policy Changes', color=color)
        ax2.plot(range(len(self.policy_changes)), self.policy_changes, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = 'tab:green'
        ax3.set_ylabel('Iteration Time (s)', color=color)
        ax3.plot(range(len(self.iteration_times)), self.iteration_times, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Learning Curve, Policy Stability, and Iteration Time')
        plt.savefig('learning_curve_and_policy_stability.png')
        plt.close(fig)
        print("Learning curve, policy stability, and iteration time plot saved as 'learning_curve_and_policy_stability.png'")
        
        
    def VI_policy_csv(self, filename='vi_policy.csv'):
        policy_list = [(self.states[i], self.policy[i]) for i in range(len(self.states))]
        df = pd.DataFrame(policy_list, columns=['State', 'Action'])
        df.to_csv(filename, index=False)
        print(f"Policy saved to {filename}")
        
        
    def getPossibleActions(self, state):
        """ Get the possible actions from a state """
        return [action for action in self.actions if self.Transition(state, action) is not None]


    def QLearning(self, num_episodes, alpha, epsilon, gamma, max_episode_length = 1000, file_name = None):
        self.Q = np.zeros((len(self.states),len(self.actions)),dtype=np.float16)

        self.P = np.zeros(len(self.states),dtype=object)

        runtimes = []
        rewards = []

        for _ in range(num_episodes):
            start_time = time.time()

            state = (0,0,0,0,0,0,0,0)
            
            episode_length = 0
            accumulated_reward = 0
            while not self.CheckGoalState(state) and episode_length < max_episode_length:
                state_idx = self.fastIndex(state)

                # Get epsilon greedy action
                if isinstance(self.P[state_idx],int) and self.P[state_idx] == 0:
                    self.P[state_idx] = np.array([*map(self.actions.index,self.getPossibleActions(state))],dtype=np.int32)
                possible_actions = self.P[state_idx]
                best_action = possible_actions[np.argmax(self.Q[state_idx,possible_actions])]

                action_idx = random.choices(possible_actions, k=1, weights = np.where(possible_actions == best_action, 1 - epsilon + epsilon/len(possible_actions), epsilon/len(possible_actions)))[0]
                action = self.actions[action_idx]

                reward = self.Reward(state, action)
                accumulated_reward += reward

                # Generate successor state

                # TODO: cache states/probability outputs of Transition
                states, probabilities = zip(*self.Transition(state, action))

                successor_state = random.choices(states, k=1, weights=probabilities)[0]
                successor_state_idx = self.fastIndex(successor_state)
                if isinstance(self.P[successor_state_idx],int) and self.P[successor_state_idx] == 0:
                    self.P[successor_state_idx] = np.array([*map(self.actions.index,self.getPossibleActions(successor_state))],dtype=np.int32)
                possible_successor_actions = self.P[successor_state_idx]

                self.Q[state_idx,action_idx] = self.Q[state_idx,action_idx] + alpha * (reward + gamma * np.max(self.Q[successor_state_idx, possible_successor_actions]) - self.Q[state_idx, action_idx])

                state = successor_state
                episode_length += 1

                if self.CheckGoalState(state):
                    accumulated_reward += self.Reward(state, ('end', None))

            rewards.append(accumulated_reward)

            elapsed = time.time() - start_time
            runtimes.append(elapsed)

        print(f"Total runtime: {sum(runtimes)}      Average runtime per episode: {sum(runtimes)/len(runtimes)}")

        if file_name:
            np.save(file_name,self.Q)

        # make a learning curve (plot of rewards?)
        plt.figure()
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Q-learning Learning curve")
        plt.plot(np.arange(len(rewards)),rewards)

        window_width = num_episodes//20
        running_sum = np.cumsum(np.pad(rewards,window_width//2,mode='edge'))
        moving_averages = (running_sum[window_width:] - running_sum[:-window_width]) / window_width
        plt.plot(np.arange(len(rewards)),moving_averages, color="red")
        plt.show()

        # make a sample trajectory after finishing q learning training  
        self.test_QLearning(0, max_episode_length)

    def test_QLearning(self, epsilon, max_episode_length = 100):
        state = (0,0,0,0,0,0,0,0)
        episode_length = 0

        accumulated_reward = 0
        while not self.CheckGoalState(state) and episode_length < max_episode_length:
            state_idx = self.fastIndex(state)
            possible_actions = np.array([*map(self.actions.index,self.getPossibleActions(state))],dtype=np.int32)
            best_action = possible_actions[np.argmax(self.Q[state_idx,possible_actions])]

            action_idx = random.choices(possible_actions, k=1, weights = np.where(possible_actions == best_action, 1 - epsilon + epsilon/len(possible_actions), epsilon/len(possible_actions)))[0]
            action = self.actions[action_idx]

            reward = self.Reward(state, action)
            accumulated_reward += reward

            self.PrintWarehouse(state, action, episode_length + 1)

            states, probabilities = zip(*self.Transition(state, action))

            successor_state = random.choices(states, k=1, weights=probabilities)[0]
            state=successor_state
            episode_length += 1
            
        if self.CheckGoalState(state):
            accumulated_reward += self.Reward(state, ('end', None))
            self.PrintWarehouse(state, ('end', None), episode_length + 1)

        print(f"Total reward earned over episode: {accumulated_reward}")

    def random(self, num_episodes, max_episode_length = 1000):
        rewards = []
        runtimes = []
        for _ in range(num_episodes):
            start_time = time.time()

            state = (0,0,0,0,0,0,0,0)
            episode_length = 0

            accumulated_reward = 0
            while not self.CheckGoalState(state) and episode_length < max_episode_length:
                action = random.choice(self.getPossibleActions(state))

                reward = self.Reward(state, action)
                accumulated_reward += reward

                states, probabilities = zip(*self.Transition(state, action))

                successor_state = random.choices(states, k=1, weights=probabilities)[0]
                state=successor_state
                episode_length += 1
                
            if self.CheckGoalState(state):
                accumulated_reward += self.Reward(state, ('end', None))
            
            elapsed = time.time() - start_time
            runtimes.append(elapsed)
            rewards.append(accumulated_reward)

        print(f"Average reward for random (possible) actions over episodes of max length {max_episode_length}: {np.average(rewards)}")
        print(f"Total runtime: {sum(runtimes)}      Average runtime per episode: {np.average(runtimes)}")
    
    def EveryVisitMonteCarlo(self, num_episodes, gamma, epsilon, max_episode_length, file_name = None):
        print(f"Episodes: {num_episodes}, Gamma: {gamma}, Epsilon: {epsilon}, Max Episode Length: {max_episode_length}")
        self.Q = np.zeros((len(self.states),len(self.actions)),dtype=np.float16)

        self.P = np.zeros(len(self.states),dtype=object)

        self.policy = {}
        self.UpdatePolicy(epsilon, self.states)

        runtimes = []
        rewards = []

        for _ in tqdm(range(num_episodes)):
            start_time = time.time()

            episode = []
            state = (0,0,0,0,0,0,0,0)
            
            episode_length = 0
            accumulated_reward = 0
            while not self.CheckGoalState(state) and episode_length < max_episode_length:
                state_idx = self.fastIndex(state)

                # Get action from epsilon-greedy policy
                action = self.policy[state]

                # Get reward
                reward = self.Reward(state, action)
                accumulated_reward += reward

                # Store episode step
                episode.append((state, action, reward))

                # Generate successor state
                states, probabilities = zip(*self.Transition(state, action))
                successor_state = random.choices(states, k=1, weights=probabilities)[0]
                state = successor_state

                episode_length += 1

                if self.CheckGoalState(state):
                    accumulated_reward += self.Reward(state, ('end', None))

            rewards.append(accumulated_reward)

            state_action_rewards_lists = {}
            encountered_state_actions = set()
            encountered_states = set()
            discounted_reward = 0
            # calculate discounted reward
            for (state, action, reward) in reversed(episode):
                discounted_reward = reward + gamma * discounted_reward

                if (state, action) not in encountered_state_actions: # Init reward storage for unseen state-action pairs 
                    state_action_rewards_lists[(state, action)] = []
                    encountered_state_actions.add((state, action))
                    encountered_states.add(state)

                # Make list of discounted rewards for each state-action pair
                state_action_rewards_lists[(state, action)] = np.append(state_action_rewards_lists[(state, action)], discounted_reward)

            # Update Q-values to mean of discounted rewards
            for (state, action) in encountered_state_actions:
                state_idx = self.fastIndex(state)
                action_idx = self.actions.index(action)
                self.Q[state_idx, action_idx] = np.mean(state_action_rewards_lists[(state, action)])
            
            self.UpdatePolicy(epsilon, encountered_states)

            elapsed = time.time() - start_time
            runtimes.append(elapsed)

        print(f"Total runtime: {sum(runtimes)}   Average runtime per episode: {sum(runtimes)/len(runtimes)}   Average reward: {sum(rewards)/len(rewards)}")

        if file_name:
            np.save(file_name,self.Q)

        # make a learning curve
        plt.figure()
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("EVMC Learning curve")
        plt.plot(np.arange(len(rewards)),rewards)

        window_width = num_episodes//20
        running_sum = np.cumsum(np.pad(rewards,window_width//2,mode='edge'))
        moving_averages = (running_sum[window_width:] - running_sum[:-window_width]) / window_width
        plt.plot(np.arange(len(rewards)),moving_averages, color="red")
        plt.show()

        # make a sample trajectory after finishing q learning training  
        self.test_EVMC(max_episode_length)

    def UpdatePolicy(self, epsilon, states):
        for state in states:
            state_idx = self.fastIndex(state)

            if isinstance(self.P[state_idx],int) and self.P[state_idx] == 0: # transition probabilities and action not yet calculated
                self.P[state_idx] = np.array([*map(self.actions.index,self.getPossibleActions(state))],dtype=np.int32) # calculate

            possible_actions = self.P[state_idx]
            best_action = possible_actions[np.argmax(self.Q[state_idx,possible_actions])]

            # chose action for policy based on epsilon-greedy policy
            action_idx = random.choices(possible_actions, k=1, weights = np.where(possible_actions == best_action, 1 - epsilon + epsilon/len(possible_actions), epsilon/len(possible_actions)))[0]
            self.policy[state] = self.actions[action_idx]

    def test_EVMC(self, max_episode_length = 100):
        cwd = os.path.dirname(os.path.abspath(__file__))
        output_file_path = os.path.join(cwd, 'agent_path.txt')
        original_stdout = sys.stdout
        with open(output_file_path, 'w') as sys.stdout:
            state = (0,0,0,0,0,0,0,0)
            episode_length = 0

            accumulated_reward = 0
            while not self.CheckGoalState(state) and episode_length < max_episode_length:
                action = self.policy[state]

                reward = self.Reward(state, action)
                accumulated_reward += reward

                self.PrintWarehouse(state, action, episode_length + 1)

                states, probabilities = zip(*self.Transition(state, action))

                successor_state = random.choices(states, k=1, weights=probabilities)[0]
                state=successor_state
                episode_length += 1
                
            if self.CheckGoalState(state):
                accumulated_reward += self.Reward(state, ('end', None))
                self.PrintWarehouse(state, ('end', None), episode_length + 1)

            print(f"Total reward earned over episode: {accumulated_reward}")
        sys.stdout = original_stdout
    
    def test_random(self, max_episode_length = 1000):
        state = (0,0,0,0,0,0,0,0)
        episode_length = 0

        accumulated_reward = 0
        while not self.CheckGoalState(state) and episode_length < max_episode_length:
            action = random.choice(self.getPossibleActions(state))

            reward = self.Reward(state, action)
            accumulated_reward += reward

            self.PrintWarehouse(state, action, episode_length + 1)

            states, probabilities = zip(*self.Transition(state, action))

            successor_state = random.choices(states, k=1, weights=probabilities)[0]
            state=successor_state
            episode_length += 1
            
        if self.CheckGoalState(state):
            accumulated_reward += self.Reward(state, ('end', None))
            self.PrintWarehouse(state, ('end', None), episode_length + 1)

        print(f"Total reward earned over episode: {accumulated_reward}")
        
        
class VISim(State):
    def __init__(self, policy_file):
        super().__init__()
        self.load_policy(policy_file)
        
    def load_policy(self, policy_file):
        policy_df = pd.read_csv(policy_file)
        
        self.policy_dict = {}
        
        for _, row in policy_df.iterrows():
            state_tuple = eval(row['State'])
            action_index = row['Action']
            self.policy_dict[state_tuple] = self.actions[action_index]


    def PrintWarehouse(self, state, action=None, action_num=None):
        for i in range(WAREHOUSE_SIZE):
            for j in range(WAREHOUSE_SIZE):
                if (i, j) == (state[0], state[1]):
                    print("A", end=" ")
                elif (i, j) in self.box_initial_locations:
                    print(f"B", end=" ")
                elif (i, j) == self.goal_location:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()
            
        print("- B1:", state[2], "- B2:", state[3], "- B3:", state[4], "- B4:", state[5], "- B5:", state[6], "action:", action, action_num)
        print("reward:", self.Reward(state, action), "\n")


    def simulate(self, start_state, steps=500):
        output_file = 'simulation_output.txt'
        state = start_state
        total_reward = 0
        
        with open(output_file, 'w') as f:
            for step in range(steps):
                if self.CheckGoalState(state):
                    f.write("Goal state reached!\n")
                    break
                
                action = self.policy_dict.get(state)
                
                if not action:
                    f.write("No action found for state: {}\n".format(state))
                    break
                
                action_num = self.actions.index(action)
                self.PrintWarehouse(state, action, action_num)
                
                reward = self.Reward(state, action)
                total_reward += reward
                f.write(self.PrintSimulatedState(state, action, action_num, reward))
                
                next_states = self.Transition(state, action)
                
                if next_states:
                    state = max(next_states, key=lambda x: x[1])[0]
                else:
                    f.write("No valid transitions from state: {}\n".format(state))
                    break

            # Print the final state and total accumulated reward in the output file
            f.write("Final state:\n")
            f.write(self.PrintSimulatedState(state, None, None, None))
            f.write(f"Total accumulated reward: {total_reward}\n")


    def PrintSimulatedState(self, state, action, action_num, reward):
        output = ""
        
        for i in range(WAREHOUSE_SIZE):
            for j in range(WAREHOUSE_SIZE):
                if (i, j) == (state[0], state[1]):
                    output += "A "
                elif (i, j) in self.box_initial_locations:
                    output += "B "
                elif (i, j) == self.goal_location:
                    output += "G "
                else:
                    output += ". "
                    
            output += "\n"
            
        output += f"- B1: {state[2]} - B2: {state[3]} - B3: {state[4]} - B4: {state[5]} - B5: {state[6]} next action: {action} {action_num}\n"
        
        if reward is not None:
            output += f"reward for next action: {reward}\n\n"
        else:
            output += "reward for next action: None\n\n"
        
        return output




#-----------------------------------------------





def VI(warehouse: State):
    print("Running VI...")
    warehouse.ValueIteration(gamma=0.99)
    warehouse.VI_plot()
    warehouse.VI_policy_csv()
    warehouse_simulation = VISim('./vi_policy.csv')
    warehouse_simulation.simulate((0, 0, 0, 0, 0, 0, 0, 0))
   
    
def QL(warehouse: State):
    print("Running QL...")
    # warehouse.ValueIteration(gamma=0.9)
    episodes = 5000
    warehouse.QLearning(episodes, alpha=0.2, epsilon=0.4, gamma=0.9, file_name=f"QLearning_{episodes}_episodes.npy")
    # warehouse.Q = np.load("QLearning_200000_episodes.npy")
    # warehouse.test_QLearning(0, 1000)
    
def Random(warehouse: State):
    print("Running random...")

    max_episode_length = 1000

    warehouse.random(1000, max_episode_length)
    warehouse.test_random(max_episode_length)
    
def EVMC(warehouse: State):
    print("Running EVMC...")
    warehouse.EveryVisitMonteCarlo(100000, gamma=0.9, epsilon=0.3, max_episode_length=500, file_name='EVMC.npy')
        

if __name__ == "__main__":
    warehouse = State()
    VI(warehouse)
    # QL(warehouse)
    # EVMC(warehouse)
    # Random(warehouse)