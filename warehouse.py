# [x, y, b1, b2, b3, b4, b5, BoxID in location]
# x and y represent the coordinates of the agent in the 10x10 warehouse
# b1-b5 represents the boxes, they can each take on values 0: in starting space, 1: Sitting in the goal space, 2: Stacked in the goal, 3: currently being carried
# boxID in current position 0: No box, boxes 1-5
# Boxes can only stack in decreasing order


class State:
    # Initialize the state
    def __init__(self):
        self.state = [[0 for _ in range(10)] for _ in range(10)]
        
        self.agent_x, self.agent_y = 0, 0 
        self.box__initial_locations = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] 
        self.goal_location = [9,9] 

        self.box_curr_locations = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] 
        self.goal_box_stack = [0, 0, 0, 0, 0]
        self.agent_has_box = False
        
    # For each value in the grid, it can denote the following:
    # 0 is empty space  
    # 1 is the agent
    # 10 is box 1 sitting in the start space
    # 11 is box 1 sitting in the goal space
    # 12 is box 1 stacked in the goal space
    # 13 is box 1 currently being carried by the agent
    # 20 is box 2 sitting in the start space
    # and so on...
    
    # Display the current state of the warehouse
    def Display(self):
    
    # Check the current box stack order is valid, if not, reset the stack
    def CheckStackOrder(self):
        for i in range(4):
            if self.goal_box_stack[i] != 0 and self.goal_box_stack[i] != i + 1:
                for i in range(4):
                    if self.goal_box_stack[i] != 0: 
                        self.goal_box_stack[i] = 0  

    # Move the agent
    def MoveAgent(self, direction):
        if direction == 'up':
            if self.agent_y > 0:
                self.agent_y -= 1
        elif direction == 'down':
            if self.agent_y < 9:
                self.agent_y += 1
        elif direction == 'left':
            if self.agent_x > 0:
                self.agent_x -= 1
        elif direction == 'right':
            if self.agent_x < 9:
                self.agent_x += 1
                
    # Move a specific box
    def MoveBox(self, box, direction):
        if direction == 'up':
            if self.box_curr_locations[box][1] > 0:
                self.box_curr_locations[box][1] -= 1
        elif direction == 'down':
            if self.box_curr_locations[box][1] < 9:
                self.box_curr_locations[box][1] += 1
        elif direction == 'left':
            if self.box_curr_locations[box][0] > 0:
                self.box_curr_locations[box][0] -= 1
        elif direction == 'right':
            if self.box_curr_locations[box][0] < 9:
                self.box_curr_locations[box][0] += 1
                
    # Get the agent location
    def GetAgentLoc(self):
        return [self.agent_x, self.agent_y]            
    
    # Get the starting location of a box
    def GetBoxLoc(self, box):
        return self.box__initial_locations[box]
    
# Initialize the warehouse and print it
state = State()
state.MoveAgent('right')
state.Display()
print(state.GetAgentLoc())
