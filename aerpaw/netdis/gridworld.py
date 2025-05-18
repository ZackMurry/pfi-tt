
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame

# Enable cuDNN benchmarking for performance (if using GPU)
torch.backends.cudnn.benchmark = True

GRID_SIZE = 15
CELL_SIZE = 50
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
NUM_OBSTACLES = 25

class Gridworld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.vision_radius = 5
        self.goals = [(GRID_SIZE - 1, GRID_SIZE - 1), (0, GRID_SIZE - 1)]
        self.obstacles = [(2, 3), (7, 8), (11, 1), (6, 4), (9, 13),
                          (3, 10), (12, 7), (0, 9), (0, 12), (4, 6),
                          (10, 3), (1, 10), (6, 9), (13, 2), (7, 6),
                          (5, 14), (11, 12), (10, 14), (8, 2), (14, 4),
                          (5, 8), (2, 10), (13, 13), (10, 9), (4, 2)]
        self.best_path_agent0 = self.a_star((0, 0), self.goals[0])
        self.best_path_agent1 = self.a_star((0, 0), self.goals[1])
        self.congestion_zones = []
        self.congestion_duration = 50
        self.congestion_timer = 0
        self.agents = [(0, 0), (0, 0)]
        self.movement_penalty = -1
        self.closeness_penalty = -10
        self.collision_penalty = -50
        self.congestion_penalty = -20
        self.goal_reward = 1000
        self.shaping_factor = 0.5
        self.best_path_bonus = 5
        self.deviation_penalty = -5

    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def get_neighbors(self, node):
        (x, y) = node
        neighbors = []
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if (nx, ny) not in self.obstacles:
                    neighbors.append((nx, ny))
        return neighbors

    def a_star(self, start, goal):
        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def update_congestion_zones(self):
        if self.congestion_timer > 0:
            self.congestion_timer -= 1
            return
        num_zones = 6
        possible_zones = list(set(self.best_path_agent0 + self.best_path_agent1))
        if possible_zones:
            self.congestion_zones = random.sample(possible_zones, min(num_zones, len(possible_zones)))
        else:
            self.congestion_zones = []
        self.congestion_timer = self.congestion_duration

    def get_obstacle_view(self, agent_position):
        view = []
        ax, ay = agent_position
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            for dx in range(-self.vision_radius, self.vision_radius + 1):
                cx, cy = ax + dx, ay + dy
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    view.append(1.0 if (cx, cy) in self.obstacles else 0.0)
                else:
                    view.append(0.0)
        return view

    def get_congestion_view(self, agent_position):
        view = []
        ax, ay = agent_position
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            for dx in range(-self.vision_radius, self.vision_radius + 1):
                cx, cy = ax + dx, ay + dy
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    view.append(1.0 if (cx, cy) in self.congestion_zones else 0.0)
                else:
                    view.append(0.0)
        return view

    def reset(self):
        self.agents = [(0, 0), (0, 0)]
        return self.get_states()

    def get_states(self):
        states = []
        for idx, pos in enumerate(self.agents):
            goal = self.goals[idx]
            base_state = [pos[0], pos[1], goal[0], goal[1]]
            obstacle_view = self.get_obstacle_view(pos)
            congestion_view = self.get_congestion_view(pos)
            states.append(base_state + obstacle_view + congestion_view)
        return states

    def step(self, actions):
        self.update_congestion_zones()
        reward = 0
        reward += self.movement_penalty * len(self.agents)
        old_distances = []
        for idx, (x, y) in enumerate(self.agents):
            goal = self.goals[idx]
            old_distances.append(abs(x - goal[0]) + abs(y - goal[1]))
        new_positions = []
        for idx, (x, y) in enumerate(self.agents):
            dx, dy = 0, 0
            action = actions[idx]
            if action == 0:
                dy = -1
            elif action == 1:
                dy = 1
            elif action == 2:
                dx = -1
            elif action == 3:
                dx = 1
            new_x, new_y = x + dx, y + dy
            if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
                new_x, new_y = x, y
            if (new_x, new_y) in self.obstacles:
                reward += self.collision_penalty
                new_x, new_y = x, y
            new_positions.append((new_x, new_y))
        self.agents = new_positions
        for idx, pos in enumerate(self.agents):
            if pos in self.congestion_zones:
                reward += self.congestion_penalty
        for idx, (x, y) in enumerate(self.agents):
            goal = self.goals[idx]
            new_distance = abs(x - goal[0]) + abs(y - goal[1])
            reward += self.shaping_factor * (old_distances[idx] - new_distance)
        for idx, pos in enumerate(self.agents):
            best_path = self.best_path_agent0 if idx == 0 else self.best_path_agent1
            if pos in best_path:
                reward += self.best_path_bonus
            else:
                congestion_view = self.get_congestion_view(pos)
                if not any(cell == 1.0 for cell in congestion_view):
                    reward += self.deviation_penalty
        manhattan_distance = abs(self.agents[0][0] - self.agents[1][0]) + abs(self.agents[0][1] - self.agents[1][1])
        if manhattan_distance < 3:
            reward += self.closeness_penalty
        for idx, pos in enumerate(self.agents):
            if pos == self.goals[idx]:
                reward += self.goal_reward
        done = (self.agents[0] == self.goals[0] and self.agents[1] == self.goals[1])
        next_states = self.get_states()
        return next_states, reward, done

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99,
                 buffer_capacity=10000, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(4)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        current_q = self.policy_net(batch_state).gather(1, batch_action)
        next_q = self.target_net(batch_next_state).max(1)[0].unsqueeze(1)
        target_q = batch_reward + self.gamma * next_q * (1 - batch_done)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
