import random
import time
import numpy as np

POSSIBLE_DIRS = ['left', 'down', 'right', 'up']
WAREHOUSE_SIZE = 10

# Example of a state: (agent_x, agent_y, b1_status, b2_status, b3_status, b4_status, b5_status, boxID of box in agent's location)

class State:
    def __init__(self):
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = (WAREHOUSE_SIZE - 1, WAREHOUSE_SIZE - 1)
        
        self.policy = {}
        self.states = []
        
        
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
    
    
    # TODO: make this function cache results
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
                state_list.append(((x + xmov, y + ymov, *state[2:]),0.05)) 
            else:
                state_list.append((state,0.05))

            # right
            direction = (originalDirection + 1) % 4 
            xmov,ymov = getMov(direction)
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                state_list.append(((x + xmov, y + ymov, *state[2:]), 0.05))
            else:
                state_list.append((state,0.05))

            # double & regular move
            xmov, ymov = getMov(originalDirection)
            xmov *= 2
            ymov *= 2
            
            if 0 <= (x + xmov) < WAREHOUSE_SIZE and 0 <= (y + ymov) < WAREHOUSE_SIZE:
                state_list.append(((x + xmov, y + ymov, *state[2:]), 0.1))
                xmov, ymov = getMov(originalDirection)
                state_list.append(((x + xmov, y + ymov, *state[2:]), 0.8))   
            else:
                xmov, ymov = getMov(originalDirection)
                state_list.append(((x + xmov, y + ymov, *state[2:]), 0.9))   
           
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

        # Test prints
        self.PrintState(state)
        self.PrintWarehouse(state)
        print("--------------------------------------")
        for s in state_list:
            self.PrintState(s[0])
            self.PrintWarehouse(s[0])

        return state_list


    def Reward(self, state, action):
        """ Reward function. Returns the reward for a given state and action.

        Args:
            state (tuple): Current state of the warehouse
            action (tuple): Action to be taken (e.g. ('move', 'left') or ('stack', 2)

        Returns:
            float: Reward for the given state and action
        """
        # Reward positive/negative for stacking based on order
        if action[0] == 'stack':
            if self.CheckStackOrder(state, int(action[1])):
                return 10
            else:
                return -50 
        
        if action[0] == 'setdown':
            if (state[0], state[1]) == self.goal_location and 3 in state[2:7]:
                return 5 

        if (state[0], state[1]) == self.goal_location: # also based on action?
                return 2

        if action[0] == 'move':
            return -1
            
        if action[0] == 'pickup':
            return 5 
    
    
    
