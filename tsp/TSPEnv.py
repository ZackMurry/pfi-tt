import numpy as np
import gymnasium as gym
from gymnasium import spaces
from or_gym import utils
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from time import sleep
from datetime import datetime
from types import SimpleNamespace

steps_log = []
dist_log = []
solved_log = []

class TSPEnv(gym.Env):
    '''
    Bi-directional connections and uniform cost

    This version of the TSP uses a sparse graph with uniform cost.
    The goal is to minimize the cost to traverse all of the nodes in the
    network. All connections are bi-directional meaning if a connection
    between nodes n and m exist, then the agent can move in either direction.
    The network is randomly generated with N nodes when the environment is
    initialized using or_gym.make(). 
    
    TSP-v0 allows repeat visits to nodes with no additional penalty beyond
    the nominal movement cost.

    Observation:
        

    Actions:
        Type: Discrete
        0: move to node 0
        1: move to node 1
        2: ...

    Action Masking (optional):
        Masks non-existent connections, otherwise a large penalty is imposed
        on the agent.

    Reward:
        Cost of moving from node to node or large negative penalty for
        attempting to move to a node via a non-existent connection.

    Starting State:
        Random node

    Episode Termination:
        All nodes have been visited or the maximimum number of steps (2N)
        have been reached.
    '''
    def __init__(self, *args, **kwargs):
        self.N = 5
        self.dist_factor = 10
        self.move_cost = -1
        self.invalid_action_cost = -100
        self.mask = False
        self.spec = SimpleNamespace(reward_threshold=1600)
        self.render_ready = False
        utils.assign_env_config(self, kwargs)

        self.locations = []
        self.min_dist = -1
        self.step_count = 0
        self.nodes = np.arange(self.N)
        self.step_limit = 2*self.N
        self.obs_dim = 1+self.N**2
        obs_space = spaces.Box(-1, self.N, shape=(self.obs_dim,), dtype=np.int32)
        self.observation_space = spaces.MultiDiscrete([self.N+1] + [self.N*2]*self.N + [100]*self.N*2)
        self.action_space = spaces.Discrete(self.N)
        self.path = []
        self.dist_sum = 0
        self.reset()
        
    def _STEP(self, action):
        done = False
        self.path.append(action)
        # Todo: reward based on comparison to minimum path instead of absolute distance (varies too much)
        dist = np.sqrt(self._node_sqdist(self.current_node, action)) / self.min_dist
        self.dist_sum += dist
        reward = -dist * self.dist_factor
        # reward = 0
        # print(f"From: {self.current_node}; To: {action}; Cost: {reward}; Stall? {action == self.current_node}; Repeat? {self.visit_log[action] > 1}")
        # print(f"Sub: {reward}")
        if action == self.current_node:
            # print('stall')
            reward -= 100
        if self.visit_log[action] >= 1:
            # print('repeat')
            reward -= 100
        # print(f"subtracting {reward}")
        self.current_node = action
        # print(action)
        if self.visit_log[self.current_node] == 0:
            # print('new node')
            reward += 100
        self.visit_log[self.current_node] += 1
            
        self.state = self._update_state()
        self.step_count += 1
        # See if all nodes have been visited
        unique_visits = sum([1 if v > 0 else 0 
            for v in self.visit_log.values()])
        if unique_visits >= self.N:
            self.path.append(self.path[0])
            dist = np.sqrt(self._node_sqdist(self.current_node, action))
            self.dist_sum += dist
            reward -= dist * self.dist_factor
            done = True
            reward += 500
        if self.step_count >= self.step_limit:
            done = True
            
        if done and self.render_ready:
            print(f"Locations: {self.locations}")
            print(f"Path: {self.path}")
            fig, ax = plt.subplots(figsize=(12,8))
            for n in range(self.N):
                pt = self.locations[n]
                ax.scatter(pt[0], pt[1], color='black', s=300)
                ax.annotate(r"$N_{:d}$".format(n), xy=(pt[0]+0.4, pt[1]+0.05), zorder=2)
            for i in range(len(self.path) - 1):
                ax.plot([self.locations[self.path[i]][0], self.locations[self.path[i+1]][0]],
                    [self.locations[self.path[i]][1], self.locations[self.path[i+1]][1]], 'bo', linestyle='solid')
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fig.savefig(f"{self.N}-solution-{current_datetime}.png")
        return self.state, reward, done, (self.step_count >= self.step_limit), {}
        
    def _node_sqdist(self, a, b):
        apt = self.locations[a]
        bpt = self.locations[b]
        dx = apt[0] - bpt[0]
        dy = apt[1] - bpt[1]
        return dx*dx + dy*dy

    def _RESET(self):
        if self.step_count > 0:
            steps_log.append(self.step_count)
            if self.step_count < 10:
                solved_log.append(1)
                dist_log.append(self.dist_sum)
            else:
                dist_log.append(1000)
                solved_log.append(0)
        self.step_count = 0
        self.dist_sum = 0
        self.current_node = np.random.choice(self.nodes)
        self.visit_log = {n: 0 for n in self.nodes}
        self.visit_log[self.current_node] += 1
        self._generate_locations()
        self.path = [self.current_node]
        self.state = self._update_state()
        return self.state, {}
        
    def _generate_locations(self):
        self.locations.clear()
        for i in range(self.N):
            self.locations.append((np.random.randint(0,100), np.random.randint(0,100)))
        self.min_dist = self.find_min_dist([self.current_node])
        # print(self.min_dist)
        

    def _update_state(self):
        visit_list = [min(self.visit_log[i], 1) for i in range(self.N)]
        state = np.array([self.current_node] + visit_list + [self.locations[i][0] for i in range(self.N)] + [self.locations[i][1] for i in range(self.N)])
        return state
            
    def _generate_coordinates(self):
        n = np.linspace(0, 2*np.pi, self.N+1)
        x = np.cos(n)
        y = np.sin(n)
        return np.vstack([x, y])

    def _get_node_distance(self, N0, N1):
        return np.sqrt(np.power(N0[0] - N1[0], 2) + np.power(N0[1] - N1[1], 2))
            
    def step(self, action):
        if self.render_ready:
            print(f"moving to {action}")
        return self._STEP(action)

    def reset(self, seed=None, options=None):
        return self._RESET()
    
    def render(self):
        if not self.render_ready:
            self.render_ready = True

    def find_min_dist(self, arr):
        low = 9999999
        low_i = -1
        unique = 0
        for i in range(self.N):
            if arr.count(i) == 0:
                md = self.find_min_dist(arr + [i]) + np.sqrt(self._node_sqdist(arr[-1], i))
                if md < low:
                    low = md
                    low_i = i
            else:
                unique += 1
        if unique == self.N:
            return np.sqrt(self._node_sqdist(arr[-1] ,arr[0]))
        return low
                

def save_steps_log():
    # print(steps_log)
    avgs = []
    num_pts = 100
    for i in range(num_pts):
        sum = 0
        for i in range(len(steps_log)//num_pts * i, len(steps_log)//num_pts * (i+1)):
            sum += steps_log[i]
        avgs.append(sum / (len(steps_log) // num_pts))
    fig, ax = plt.subplots()
    ax.scatter(range(num_pts), avgs)
    fig.savefig('steps_log.png')

def save_dist_log():
    # print(steps_log)
    avgs = []
    num_pts = 100
    for i in range(num_pts):
        sum = 0
        for i in range(len(dist_log)//num_pts * i, len(dist_log)//num_pts * (i+1)):
            sum += dist_log[i]
        avgs.append(sum / (len(dist_log) // num_pts))
    fig, ax = plt.subplots()
    ax.scatter(range(num_pts), avgs)
    fig.savefig('dist_log.png')

def save_solved_log():
    # print(steps_log)
    avgs = []
    num_pts = 100
    for i in range(num_pts):
        sum = 0
        for i in range(len(solved_log)//num_pts * i, len(solved_log)//num_pts * (i+1)):
            sum += solved_log[i]
        avgs.append(sum / (len(solved_log) // num_pts))
    fig, ax = plt.subplots()
    ax.scatter(range(num_pts), avgs)
    fig.savefig('solved_log.png')
