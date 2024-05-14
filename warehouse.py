class Warehouse:
    def __init__(self, agent_x, agent_y, goal_location, box_initial_locations, warehouse_size):
        """Initializes the warehouse environment.

        Args:
            agent_x (int): X coordinate of the agent
            agent_y (int): Y coordinate of the agent
            goal_location ( Tuple[int, int] ): Coordinates of the goal location (x, y)
            box_initial_locations (List[Tuple[int, int]]): List of coordinates of the boxes [(x1, y1), (x2, y2), ...] (max 5 boxes)
            warehouse_size (int): Size of the warehouse (square)
        """
        self.state = [[0 for _ in range(warehouse_size)] for _ in range(warehouse_size)]
        
        self.agent_x, self.agent_y = agent_x, agent_y 
        self.box_initial_locations = box_initial_locations 
        self.goal_location = goal_location 
        self.box_states = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for i in range(5):
            self.state[self.box_initial_locations[i][0]][self.box_initial_locations[i][1]] = i + 1
    
    
    def PrintWarehouse(self):
        """Prints the warehouse state in a grid format.
        """
        print('Warehouse:')
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                print(f'{self.state[i][j]:4}', end = ' ')
            print()
        
        
    def GetAgentLoc(self):
        """Returns the location of the agent in the warehouse.

        Returns:
            Tuple: Coordinates of the agent
        """
        return self.agent_x, self.agent_y
    
    
    def GetCellContents(self, x, y):
        """Check the value of the cell in the warehouse. 
            If the cell is empty, return 0. 
            If the cell has a box, return the box ID. 
            If the cell has a stack of boxes, return the stack of boxes.

        Args:
            x, y (int): Coordinates of the cell.

        Returns:
            int: Value of the cell
        """
        return self.state[x][y] 
    
    
    def GetBoxState(self, box_id):
        """Returns the state of the box.

        Args:
            box_id (int): ID of the box

        Returns:
            int: State of the box
        """
        return self.box_states[box_id - 1]
    
    
    def GetInitialBoxLoc(self, box_id):
        """Returns the initial location of the box.

        Args:
            box_id (int): ID of the box

        Returns:
            Tuple: Initial coordinates of the box
        """
        return self.box_initial_locations[box_id - 1]
    
    
    def GetWarehouseSize(self):
        """Returns the size of the warehouse.

        Returns:
            int: Size of the warehouse
        """
        return len(self.state)
    
    
    def SetBoxState(self, box_id, state):
        """Sets the state of the box.

        Args:
            box_id (int): ID of the box
            state (int): State of the box
        """
        self.box_states[box_id - 1] = state
    
    
    def StackBox(self, box_id):
        """Stacks a given box in the goal location

        Args:
            box_id (int): ID of the box to be stacked

        Returns:
            bool: True if the box was successfully stacked, False otherwise
        """
        if self.GetCellContents(self.goal_location[0], self.goal_location[1]) == 0:
            self.state[self.goal_location[0]][self.goal_location[1]] = box_id
            return True
        else:
            stack_value = self.GetCellContents(self.goal_location[0], self.goal_location[1])
            if box_id < stack_value % 10:
                self.state[self.goal_location[0]][self.goal_location[1]] = stack_value * 10 + box_id
                return True
            else:
                self.state[self.goal_location[0]][self.goal_location[1]] = box_id
                return False
    
    
    def MoveAgent(self, cordinates):
        """Moves the agent to the given cordinates.

        Args:
            cordinates (Tuple[int, int]): Coordinates to move the agent to.

        Returns:
            Tuple[int, int]: New coordinates of the agent
        """
        self.agent_x, self.agent_y = cordinates
        return self.agent_x, self.agent_y
    
    
    
    
    
    
    

env = Warehouse(agent_x=0, 
                agent_y=0, 
                goal_location=(9, 9), 
                box_initial_locations=[(1, 5), (8, 2), (6, 4), (3, 7), (4, 1)])

env.PrintWarehouse()
# print(env.GetAgentLoc())

# print(env.GetAgentLoc()[0])

# print(env.CheckCell(9, 3))

env.StackBox(5)
env.PrintWarehouse()
env.StackBox(4)
env.PrintWarehouse()
env.StackBox(3)
env.PrintWarehouse()
env.StackBox(2)
env.PrintWarehouse()
env.StackBox(1)
env.PrintWarehouse()

#Box IDs correlate to the weight of the box. Box 5 is heavier than box 1. This function, given a box ID, is to stack the given box ID in the goal spot as ints. A stack is represented as an int. If I were to stack the box 3 and there is no stack, the value of the goal space becomes 3. If I were to stack box 2 in the stack, it becomes 32. This works because the box 2 is lighter than box 3. But if I stack box 4 now, the ffunction will check to see if this is valid. Because box 4 us heavier than all the boxes under it, the stack resets.