import random
import time
import numpy as np
# [x, y, b1, b2, b3, b4, b5, BoxID in location]
# x and y represent the coordinates of the agent in the 10x10 warehouse
# b1-b5 represents the boxes, they can each take on values 0: in starting space, 1: Sitting in the goal space, 2: Stacked in the goal, 3: currently being carried
# boxID in current position 0: No box, boxes 1-5
# Boxes can only stack in decreasing order

POSSIBLE_DIRS = ['left', 'down', 'right', 'up']
WAREHOUSE_SIZE = 10

class State:
    def __init__(self):
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = (WAREHOUSE_SIZE - 1, WAREHOUSE_SIZE - 1)
        
        self.policy = {}
        self.states = []
        
        
    def CalculateAllStates(self):
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
                                    
    
    # Check if the boxes are stacked in the correct order
    def CheckStackOrder(self, state, box):
        if state[box + 2] == 2:  # Check if the box is already stacked
            return False
        current_stack = [i for i in range(5) if state[i + 2] == 2]
        if not current_stack:  # No boxes stacked, any box can be stacked
            return True
        return all(box < stacked_box for stacked_box in current_stack)

    
    def PrintState(self, state):    
        print("Agent Location: ", state[0], state[1])
        print("Boxes: ", state[2:7])
        print("BoxID in current location: ", state[7])
        
        
    def PrintWarehouse(self, state):
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
    # Given a state and action, return a tuple of states with probabilities of each state
    def Transition(self, state, action):
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
                # if (x + xmov, y + ymov, *state[2:]) doesn't work as expected, fix this & other occurences
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
                if self.CheckStackOrder(state, int(action[1])):
                    # Stack the box
                    new_state[int(action[1])+2] = 2
                else:
                    for i in range(5):  # Collapse the stack if order is incorrect
                        if state[i + 2] == 2:
                            new_state[i + 2] = 1
                state_list.append((tuple(new_state), 1))
                
        # Calculate possible stack results
        elif action[0] == "setdown":
            # Not in goal or no box carried
            if (state[0], state[1]) != self.goal_location or 3 not in state[2:7]:
                return None
            new_state = list(state)
            box_idx = state[2:7].index(3) + 2 
            new_state[box_idx] = 1
            state_list.append((tuple(new_state), 1))
        
        elif action[0] == "pickup":
            # double check later to make sure state[state[7]+2] is right for checking that the box that should be here has been picked up
            
            # no box initially starts here, the box that's supposed to be here has been picked up, or we are already carrying a box# double check
            if state[7] == 0 or state[state[7]+2] != 0 or 3 in state[2:7]:
                return None
            
            new_state = list(state)
            new_state[state[7]+1] = 3
            state_list.append((tuple(new_state), 1))

        self.PrintState(state)
        self.PrintWarehouse(state)

        print("--------------------------------------")

        for s in state_list:
            self.PrintState(s[0])
            self.PrintWarehouse(s[0])

        return state_list


    # Given a state and action, return the reward value of the action
    def Reward(self, state, action):
        # Reward positive/negative for stacking based on order
        if action[0] == 'stack':
            if self.CheckStackOrder(state, int(action[1])):
                return 10
            else:
                return -50 
        
        # Positive reward for setting down in goal location
        if action[0] == 'setdown':
            if (state[0], state[1]) == self.goal_location and 3 in state[2:7]:
                return 5 

        # Positive reward for being in goal location
        if (state[0], state[1]) == self.goal_location: # also based on action?
                return 2

        # Penalty for moving
        if action[0] == 'move':
            return -1
            
        # Small positive reward for picking up a box
        if action[0] == 'pickup':
            return 5 
    

    
    # def ValueIteration(self):
    #     self.CalculateAllStates()
    #     self.V = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
    #     self.Q =  np.zeros((len(self.states), len(self.actions))).astype('float32')
    #     max_trials = 1000
    #     epsilon = 0.00001
    #     initialize_bestQ = -10000
    #     curr_iter = 0
        
    #     bestAction = np.full((len(self.states)), -1)
        
    #     start_time = time.time()
        
    #     while curr_iter < max_trials:
    #         max_residual = 0
    #         curr_iter += 1
    #         print("Current Iteration = ", curr_iter)
            
    #         # Loop over states to calculate values
    #         for state in self.states:
    #             if state

        
    #     end_time = time.time()
        
    #     print("Time taken to solve (seconds): ", end_time - start_time)
    
    
    # def QValueCalculation(self, state, action):
    #     qAction = 0
    #     succesorStates = self.Transition(state, action)


warehouse = State()

warehouse.Transition((9, 9, 1, 2, 1, 2, 2, 0), ("stack", 2))

print()