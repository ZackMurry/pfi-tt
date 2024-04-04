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
        self.dist_factor = 1/3
        self.move_cost = -1
        self.invalid_action_cost = -100
        self.mask = False
        self.spec = SimpleNamespace(reward_threshold=1300)
        self.render_ready = False
        utils.assign_env_config(self, kwargs)

        self.locations = []
        self.step_count = 0
        self.nodes = np.arange(self.N)
        self.step_limit = 2*self.N
        self.obs_dim = 1+self.N**2
        obs_space = spaces.Box(-1, self.N, shape=(self.obs_dim,), dtype=np.int32)
        # if self.mask:
        self.observation_space = spaces.MultiDiscrete([self.N+1] + [self.N*2]*self.N + [100]*self.N*2)
        # self.observation_space = spaces.Dict({
        #     # "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.int8),
        #     # "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.int8),
        #     "state": obs_space,
        #     "x_coords": spaces.MultiDiscrete([100]*self.N),
        #     "y_coords": spaces.MultiDiscrete([100]*self.N)
        # })
        # print(self.observation_space.shape)
        # else:
        #     self.observation_space = obs_space
        self.action_space = spaces.Discrete(self.N)
        self.path = []
        self.dist_sum = 0
        self.reset()
        
    def _STEP(self, action):
        # print(f"stepping to {action}")
        done = False
        # connections = self.node_dict[self.current_node]
        # Invalid action
        # if action not in connections:
        #     reward = self.invalid_action_cost
        #     if self.render_ready:
        #         print('Invalid action!')
        # Move to new node
        # else:
        self.path.append(action)
        dist = np.sqrt(self._node_sqdist(self.current_node, action))
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
            done = True
            # print(f"DONE! {self.path}")
            # print("DONE!")
            reward += 1000
        if self.step_count >= self.step_limit:
            self.path.append(path[0])
            dist = np.sqrt(self._node_sqdist(self.current_node, action))
            self.dist_sum += dist
            reward -= dist * self.dist_factor
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

        # if done:
        #     print(f"Num steps: {self.step_count}")
        # self.plot_network()
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

    def _update_state(self):
        # node_connections = self.adjacency_matrix.copy()
        # Set value to 1 for existing, un-visited nodes
        # Set value to -1 for existing, visited nodes
        # Set value to 0 if connection doesn't exist
        # visited = np.array([bool(min(v, 1))
        #     for v in self.visit_log.values()])
        # node_connections[:, visited] = -1
        # node_connections[np.where(self.adjacency_matrix==0)] = 0

        # connections = node_connections.flatten().astype(int)
        # obs = npi.hstack([self.current_node, connections], dtype=np.int32)
        # if self.mask:
        #     mask = node_connections[self.current_node]
        #     # mask = np.array([1 if c==1 and v==0 else 0 
        #     #     for c, v in zip(cons_from_node, self.visit_log.values())])
        #     state = {
        #         # "action_mask": mask,
        #         # "avail_actions": np.ones(self.N, dtype=np.uint8),
        #         "state": self.current_node,
        #     }
        # else:
        #     state = obs.copy()
        # state = {
        #     "state": self.current_node,
        #     "x_coords": np.array([self.locations[i][0] for i in range(self.N)]),
        #     "y_coords": np.array([self.locations[i][1] for i in range(self.N)])
        # }
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
            
    def plot_network(self, offset=(0.02, 0.02)):
        # plt.axes.scatter()
        # plt.axes.scatter(coords[0], coords[1], s=40)
        # plt.axes()
        # plt.scatter()
        plt.cla()
        # self.ax.clear()
        # self.ax.scatter(self.coords[0], self.coords[1], s=40)
        # for n, c in self.node_dict.items():
        for n in range(self.N):
            # for k in c:
            #     pt = self.locations[n]
            #     line = np.vstack([pt[0], pt[1]])
            #     dis = self._get_node_distance(line[0], line[1])
            #     # dis = np.sqrt(np.power(line[0, 0] - line[1, 0], 2) + 
            #     #               np.power(line[0, 1] - line[1, 1], 2))
            #     self.ax.plot(line[:,0], line[:,1], c='black', zorder=-1)
                # ax.arrow(line[0, 0], line[0, 1], line[1, 0], line[1, 1])
            # if self.current_node == n:
            #     self.ax.annotate(r"Agent".format(n), xy=(self.coords[:,n]+offset), zorder=2)
            # else:
            # print(self.coords[0,n], self.coords[1,n])
            # print(f"{self.current_node} vs {n}")
            clr = 'green' if self.visit_log[n] != 0 else 'red'
            if self.current_node == n:
                clr = 'blue'
            # print(clr)
            pt = self.locations[n]
            self.ax.annotate(r"$N_{:d}$".format(n), xy=(pt[0]+offset[0], pt[1]+offset[1]), zorder=2)
            self.ax.scatter(pt[0], pt[1], color=clr, s=400)
            
        # self.ax.xaxis.set_visible(False)
        # self.ax.yaxis.set_visible(False)
        plt.pause(1)

    def step(self, action):
        if self.render_ready:
            print(f"moving to {action}")
        return self._STEP(action)

    def reset(self, seed=None, options=None):
        return self._RESET()
    
    def render(self):
        if not self.render_ready:
        #     print('first render!')
        #     # self.reset()
        #     plt.ion()
        #     self.fig, self.ax = plt.subplots(figsize=(12,8))
        #     self.coords = self._generate_coordinates()
        #     self.ax.scatter(self.coords[0], self.coords[1], s=40)
            self.render_ready = True
        #     plt.pause(3)
        # print('render!')
        #self.plot_network()


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
