import random
# [x, y, b1, b2, b3, b4, b5, BoxID in location]
# x and y represent the coordinates of the agent in the 10x10 warehouse
# b1-b5 represents the boxes, they can each take on values 0: in starting space, 1: Sitting in the goal space, 2: Stacked in the goal, 3: currently being carried
# boxID in current position 0: No box, boxes 1-5
# Boxes can only stack in decreasing order

# For each value in the grid, it can denote the following:
# 0 is empty space  
# 1 is the agent
# 10 is box 1 sitting in the start space
# 11 is box 1 sitting in the goal space
# 12 is box 1 stacked in the goal space
# 13 is box 1 currently being carried by the agent
# 20 is box 2 sitting in the start space
# and so on...


possible_dirs = ['up', 'right', 'down', 'left']


class State:
    # Initialize the state and the following
    def __init__(self):
        self.box_initial_locations = [(3, 5), (1, 8), (5, 4), (9, 1), (7, 2)]
        self.goal_location = [self.warehouse_size - 1] * 2
        self.warehouse_size = 10

        # should be in agent
        self.agent_x, self.agent_y = 0, 0 
        self.goal_box_state = [0, 0, 0, 0, 0]
        self.agent_has_box = False
        
        # States of boxes
        self.box_states = [0, 0, 0, 0, 0]


    # Creates a tuple that represents the state of boxes 
    def BoxStates(self):
        """(status of box1, status of box2...)

        Returns:
            Tuple: State representation tuple of the warehouse
        """
        # The tuple returned should be like this:
        # (status of box1, status of box2...)
        # 0 is box sitting in the start space
        # 1 is box sitting in the goal space
        # 2 is box stacked in the goal space
        # 3 is box currently being carried by the agent
        
        box_states = []
        
                
        
        
        
    # Display the state representation tuple of the warehouse
    def Display(self):
        print(self.StateRepresentation())
    
    
    # Check the current box stack order is valid, if not, reset the stack
    def CheckStackOrder(self):
        for i in range(4):
            if self.goal_box_stack[i] != 0 and self.goal_box_stack[i] != i + 1:
                for i in range(4):
                    if self.goal_box_stack[i] != 0: 
                        self.goal_box_stack[i] = 0  


    # Move the agent, given cordinates and desired action, output new cordinates 
    def MoveAgent(self, current_position, action):
        # Represent the action as an integer
        direction = possible_dirs.index(action)
        
        # Use random to determine if move is successful
        random_num = random.random()
        
        doubleMove = False
        
        # Check for failed moves
        if random_num < 0.05:
            # Slide left
            direction = (direction - 1) % 4
        elif random_num < 0.1:
            # Slide right
            direction = (direction + 1) % 4
        elif random_num < 0.2:
            # Move double
            doubleMove = True
        
        # Move the agent in the desired direction if allowed
        xmov = [0,1,0,-1][direction]
        ymov = [1,0,-1,0][direction]

        if doubleMove:
            xmov *= 2
            ymov *= 2

        # Ensure the agent stays within the warehouse
        # May be more performant to do ifs vs max of min
        xpos = max(min(current_position[0] + xmov, 0), self.warehouse_size - 1)
        ypos = max(min(current_position[1] + ymov, 0), self.warehouse_size - 1)

        return (xpos, ypos)
                

    # Don't take invalid pickup/stack/setdown actions
    def TakeAction(self, state, action):
        if action in possible_dirs:
            location = (state[0], state[1])
            new_position = self.MoveAgent(action, location)
            
            if new_position in self.box_initial_locations:
                boxID_in_location = self.box_initial_locations.index(new_position) + 1
            else:
                boxID_in_location = 0
            
            return (new_position[0], new_position[1], state[2], state[3], state[4], state[5], state[6], boxID_in_location)
        
        elif action == 'pickup':
            self.agent_has_box = True
            state_list = list(state)
            # Set the box to being carried,  does not check if box is in location or already being carried
            state_list[7] = 3
            return tuple(state_list)
            
        elif action == 'stack':
            self.CheckStackOrder()
            
        elif action == 'setdown':
            pass
           
    
    # Get the starting location of a box
    def GetBoxLoc(self, box):
        return self.box_initial_locations[box]




    def cached_function():
        cache = {}

        def __function(state, other_parameters):
            
            if state not in cache:
                # do anything needed with state, like pulling possible actions
                cache[state] = # something

            return cache[state]

        return __function

    function_that_uses_cache = cached_function()


# ---------------------------- Agent Class ----------------------------


class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        
        
    # Get the agent's location
    def GetLocation(self):
        return [self.x, self.y]
    
    
# Initialize the warehouse and print it
state = State()
agent = Agent()

state.MoveAgent('right')

