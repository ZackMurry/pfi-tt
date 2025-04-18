import torch
import numpy as np
from gridworld import Gridworld, DQN  # Assuming your environment & model code are in gridworld.py

def get_path(start, end, obstacles):
    # Create and configure the environment
    env = Gridworld()
    
    # Set both agents to the same start, but only the first agent will be used
    env.agents = [start, (0, 0)]          # Second agent is dummy
    env.goals = [end, (0, 0)]             # Second goal is dummy
    env.obstacles = obstacles             # Replace obstacles with the provided list
    
    # Recompute best paths since obstacles changed
    env.best_path_agent0 = env.a_star(start, end)
    env.best_path_agent1 = env.a_star((0, 0), (0, 0))  # Dummy

    # Reset the environment and get the initial state for the agent
    state = env.reset()[0]  # For a single agent, take the first state

    # Determine state dimension and action dimension
    state_dim = len(state)
    action_dim = 4  # up, down, left, right

    # Load the trained model (ensure that trained_model.pth is provided)
    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load("agent_0_trained.pth", map_location=torch.device("cpu")))
    model.eval()

    # Run the simulation until the goal is reached (or a maximum step count)
    path = [start]
    done = False
    max_steps = 1000
    current_state = state
    while not done and len(path) < max_steps:
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        next_state, reward, done = env.step([action, 0])  # Second agent is dummy
        current_state = next_state[0]
        path.append(env.agents[0])
    return path

if __name__ == '__main__':
    #Start and End
    start = (14, 14)
    end = (5, 5)

    #Obstacle list
    obstacles = [(2, 3), (7, 8), (11, 1), (6, 4), (9, 13),
    (3, 10), (12, 7), (0, 9), (0, 12), (4, 6),
    (10, 3), (1, 10), (6, 9), (13, 2), (7, 6),
    (5, 14), (11, 12), (10, 14), (8, 2), (14, 4),
    (5, 8), (2, 10), (13, 13), (10, 9), (4, 2)]
    obstacles = [(1,1),(2,2), (7,7)]
    path = get_path(start, end, obstacles)
    print("Computed Path (as grid waypoints):")
    print(path)
