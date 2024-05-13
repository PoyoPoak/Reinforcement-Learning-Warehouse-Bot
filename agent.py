import random
import warehouse

MOVEMENT = ['up', 'right', 'down', 'left']
WAREHOUSE_SIZE = 10

class Agent:
    def __init__(self, warehouse):
        self.warehouse = warehouse
        
        self.agent_x, self.agent_y = self.warehouse.GetAgentLoc()
        self.agent_has_box = False
        
    def MoveAgent(self, action):
        """Move the agent in the warehouse based on the action provided. 
        Check for failed moves and ensure the agent stays within the warehouse.

        Args:
            action (str): Action to be taken by the agent

        Returns:
            Tuple: New position of the agent
        """
        # Represent the action as an integer
        direction = MOVEMENT.index(action)
        
        random_num = random.random()
        doubleMove = False
        
        # Check for failed moves
        if random_num < 0.05: # left
            print("Slipping left...")
            direction = (direction - 1) % 4
        elif random_num < 0.1: # right
            print("Slipping right...")
            direction = (direction + 1) % 4
        elif random_num < 0.2: # double move
            print("Double move...")
            doubleMove = True
        
        # Move in desired direction if able
        xmov = [0, 1, 0, -1][direction]
        ymov = [-1, 0, 1, 0][direction]

        if doubleMove:
            xmov *= 2
            ymov *= 2

        current_position = self.warehouse.GetAgentLoc()
        
        # Ensure the agent stays within the warehouse
        xpos = max(0, min(WAREHOUSE_SIZE - 1, current_position[0] + xmov))
        ypos = max(0, min(WAREHOUSE_SIZE - 1, current_position[1] + ymov))
        
        self.warehouse.MoveAgent((xpos, ypos))
        return (xpos, ypos)

    
    # # Don't take invalid pickup/stack/setdown actions
    # def TakeAction(self, state, action):
    #     if action in possible_dirs:
    #         location = (state[0], state[1])
    #         new_position = self.MoveAgent(action, location)
            
    #         if new_position in self.box_initial_locations:
    #             boxID_in_location = self.box_initial_locations.index(new_position) + 1
    #         else:
    #             boxID_in_location = 0
            
    #         return (new_position[0], new_position[1], state[2], state[3], state[4], state[5], state[6], boxID_in_location)
        
    #     elif action == 'pickup':
    #         self.agent_has_box = True
    #         state_list = list(state)
    #         # Set the box to being carried,  does not check if box is in location or already being carried
    #         state_list[7] = 3
    #         return tuple(state_list)
            
    #     elif action == 'stack':
    #         self.CheckStackOrder()
            
    #     elif action == 'setdown':
    #         pass








env = warehouse.Warehouse(agent_x=5,
                          agent_y=9,  
                          goal_location=(9, 9), 
                          box_initial_locations=[(1, 5), (8, 2), (6, 4), (3, 7), (4, 1)],
                          warehouse_size=WAREHOUSE_SIZE)
agent = Agent(env)
env.PrintWarehouse()
print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())
# agent.MoveAgent('up')
# print(env.GetAgentLoc())



