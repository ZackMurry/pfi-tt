import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
# from transform import flatten as flatten_discrete
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from time import sleep
from datetime import datetime
from types import SimpleNamespace

steps_log = []
dist_log = []
solved_log = []

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def get_route_from_action(action):
    route = []
    route.append(action // factorial(4))
    action %= factorial(4)
    route.append(action // factorial(3))
    action %= factorial(3)
    route.append(action // factorial(2))
    action %= factorial(2)
    route.append(action // factorial(1))
    action %= factorial(1)
    for i in range(5):
        if not i in route:
            route.append(i)
            break
    return route

class HeuristicTSPEnv(gym.Env):
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
        self.t = 0
        self.x = 0
        self.y = 0
        self.MAX_T = 100
        self.MAX_NODES = 20
        self.MAX_QUEUE = 5
        self.nodes_proposed = 0
        self.use_dataset = False
        self.step_count = 0
        self.spec = SimpleNamespace(reward_threshold=100)
        self.render_ready = False
        self.MAX_X = 20
        self.MAX_Y = 20
        self.request = None
        # self.request = {
        #     "id": 0,
        #     "x": 0,
        #     "y": 0,
        #     "deadline": 0
        # }
        # customers is sorted in order of ID
        self.customers = [
            {
                "id": 0,
                "x": 3,
                "y": 5,
                "deadline": 10
            },
            {
                "id": 1,
                "x": 2,
                "y": 1,
                "deadline": 15
            },
            {
                "id": 2,
                "x": 7,
                "y": 10,
                "deadline": 30
            },
            {
                "id": 3,
                "x": 5,
                "y": 8,
                "deadline": 30
            },
        ] # index=id, x, y, deadline
        self.planned_route = []
        self.depot = {
            "x": 0,
            "y": 0
        }

        self.step_limit = 30

        self.observation_space = spaces.Dict({
            "t": spaces.Discrete(self.MAX_T),
            "request": spaces.Dict({
                "id": spaces.Discrete(self.MAX_QUEUE),
                "x": spaces.Discrete(self.MAX_X),
                "y": spaces.Discrete(self.MAX_Y),
                "deadline": spaces.Discrete(self.MAX_T)
            }),
            "customers": spaces.Dict({
                "id": spaces.MultiDiscrete([self.MAX_QUEUE]*self.MAX_QUEUE),
                "x": spaces.MultiDiscrete([self.MAX_X]*self.MAX_QUEUE),
                "y": spaces.MultiDiscrete([self.MAX_Y]*self.MAX_QUEUE),
                "deadline": spaces.MultiDiscrete([self.MAX_T]*self.MAX_QUEUE)
            }),
            "planned_route": spaces.MultiDiscrete([self.MAX_NODES]*self.MAX_QUEUE)
        })

        self.action_space = spaces.Discrete(2 * 5 * 4 * 3 * 2)
        # self.action_space = spaces.Box(np.array([0,0]), np.array([2,3]), dtype=int)

        self._plan_route()
        self.reset()
    


    def _STEP(self, action):
        done = False
        raw_action = action
        self.step_count += 1
        
        # print(f"Action: {action}")
        # print(unflatten(self.action_space, action))
        # return {}, 0, False, False, {}
        accept = action & 1
        action //= 2
        route = get_route_from_action(action)
        print(f"Accept: {accept}, route: {route}, raw: {raw_action}")
        # route = get_route_from_action(119)
        # print(f"route: {route}")
        
        reward = 0
        if accept == 1:
            if self.request == None:
                reward -= 3
            else:
                reward += 1
                self.customers.append(self.request)
                self.request = None


        
        # Check for route longer than available customers
        # i.e., non-served routes need id=0=null
        for i in range(len(route) - len(self.customers)):
            if route[-i-1] != 0:
                reward -= 1

        route = route[0:len(self.customers)+accept]

        # Check for duplicates
        for i in range(len(route)):
            for j in range(i+1, len(route)):
                if route[i] == route[j]:
                    route[j] = -1
        
        for i in range(len(route)):
            if route[len(route)-i-1] == -1:
                # print('Duplicate destination')
                reward -= 1
                route.pop(len(route) - i - 1)
                i -= 1
                    
        print(f"sanitized route: {route}")

        for cust in self.customers:
            if not cust in route:
                reward -= 3
        
        if reward >= 0:
            print('good route!')

        # Validate feasibility
        # time = self.t
        # vx = self.x
        # vy = self.y
        # for dest in route:
        #     for i in range(len(self.customers)):
        #         cust = self.customers[i]
        #         if cust['id'] == dest:
        #             dt = self._get_travel_time(self.x, self.y, cust)
        #             if time + dt > cust['deadline']:
        #                 reward -= 2
        #                 # Drop customer
        #                 # self.customers.pop(i)
        #                 # route.remove(i)
        #                 # i -= 1
        #             break
        
        # if done and self.use_dataset:
        #     print(f"Path: {self.path}")
        #     print("Optimal: 0 2 1 4 3")
        #     print(f"Distance: {self.dist_sum}")

        if done and self.render_ready:
            print(f"Locations: {self.locations}")
            print(f"Path: {self.path}")
            print(f"Efficiency: {self.dist_sum / self.min_dist}; {self.dist_sum} vs. {self.min_dist}")
            fig, ax = plt.subplots(figsize=(12,8))
            for n in range(self.N):
                pt = self.locations[n]
                clr = 'green' if n == 0 else 'black'
                ax.scatter(pt[0], pt[1], color=clr, s=300)
                ax.annotate(r"$N_{:d}$".format(n), xy=(pt[0]+0.4, pt[1]+0.05), zorder=2)
            for i in range(len(self.path) - 1):
                ax.plot([self.locations[self.path[i]][0], self.locations[self.path[i+1]][0]],
                    [self.locations[self.path[i]][1], self.locations[self.path[i+1]][1]], 'bo', linestyle='solid')
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fig.savefig(f"{self.N}-solution-{current_datetime}.png")
        return self.state, reward, done, (self.step_count >= self.step_limit), {}
    
    def _node_dist(self, a, b):
        return np.sqrt(self._node_sqdist(a, b))

    def _node_dist_manhattan(self, a, b):
        apt = self.locations[a]
        bpt = self.locations[b]
        dx = apt[0] - bpt[0]
        dy = apt[1] - bpt[1]
        return abs(dx) + abs(dy)

    def _node_sqdist(self, a, b):
        apt = self.locations[a]
        bpt = self.locations[b]
        dx = apt[0] - bpt[0]
        dy = apt[1] - bpt[1]
        return dx*dx + dy*dy

    def _RESET(self):
        np.random.seed()
        self.t = 0
        self.x = 0
        self.y = 0
        self.nodes_proposed = 0
        # self.customers = []
        self.request = self._generate_request()
        # self.generate_1d_distance_matrix()
        self._plan_route()
        self.state = self._update_state()
        return self.state
        
    def _generate_request(self):
        self.nodes_proposed += 1
        return {
            "id": self.nodes_proposed,
            "x": np.random.randint(0, self.MAX_X),
            "y": np.random.randint(0, self.MAX_Y),
            "deadline": np.random.randint(self.t, self.MAX_T)
        }
        

    def _update_state(self):
        self._plan_route()
        req = self.request
        if req == None:
            req = {
                "id": 0, # id of 0 => null
                "x": 0,
                "y": 0,
                "deadline": 0
            }
        custs = {
            "id": [],
            "x": [],
            "y": [],
            "deadline": []
        }
        i = 1
        for cust in self.customers:
            custs['id'].append(i)
            custs['x'].append(cust['x'])
            custs['y'].append(cust['y'])
            custs['deadline'].append(cust['deadline'])
            i += 1
        
        while len(custs['id']) < self.MAX_QUEUE:
            custs['id'].append(0) # id of 0 => null
            custs['x'].append(0)
            custs['y'].append(0)
            custs['deadline'].append(0)

        route = self.planned_route
        while len(route) < self.MAX_QUEUE:
            route.append(0)

        state = {
            "t": self.t,
            "request": self.request,
            "customers": custs,
            "planned_route": self.planned_route
        }
        print(f'state: {state}')
        return state
    
    def _plan_route(self):
        x = self.depot.get('x')
        y = self.depot.get('y')
        visited = []
        to_visit = []
        for i in range(len(self.customers)):
            to_visit.append({
                "id": self.customers[i].get('id'),
                "x": self.customers[i].get('x'),
                "y": self.customers[i].get('y'),
                "deadline": self.customers[i].get('deadline')
            })
        
        # print(f"Customers: {to_visit}")

        time = 0
        # Go to the next closest customer every time
        while len(to_visit) > 0 and time < self.MAX_T:
            min_t = self.MAX_T + 1
            min_t_id = -1
            min_t_i = -1
            for i in range(len(to_visit)):
                this_t = self._get_travel_time(x, y, self.customers[to_visit[i].get('id')])
                if this_t < min_t:
                    min_t = this_t
                    min_t_id = to_visit[i].get('id')
                    min_t_i = i
            cx = self.customers[min_t_id].get('x')
            cy = self.customers[min_t_id].get('y')
            print(f"Min t: {min_t}s from ({x}, {y}) to Customer {min_t_id} at ({cx}, {cy})")
            x = cx
            y = cy
            time += min_t
            visited.append(min_t_id)
            to_visit.pop(min_t_i)
        
        # print(f"Total time: {time}")
        return visited

    def _get_travel_time(self, x, y, customer):
        return abs(customer.get('x') - x) + abs(customer.get('y') - y)      

    def generate_1d_distance_matrix(self):
        matrix = []
        # max_dist = 100 * np.sqrt(2)
        for i in range(self.N):
            for j in range(i+1, self.N):
                matrix.append(self._node_dist_manhattan(i, j))
        # print(f"matrix: {matrix}")
        self.dist_matrix = matrix

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
        obs = self._RESET()
        return obs, {}
        # return flatten(self.observation_space, obs), {}
    
    def render(self):
        if not self.render_ready:
            self.render_ready = True

    def read_data(self):
        f = open('data/five_d.txt', 'r')
        self.dist_matrix = []
        in_matrix = []
        lines_list = f.readlines()
        i = 0
        for line in lines_list:
            print(line)
            new_line = line[0:-1]
            nums = line.split(' ')
            i += 1
            j = 0
            for num in nums:
                if num == '':
                    continue
                j += 1
                if num and j > i:
                    if int(round(float(num))) == 0:
                        print(f"i: {i}; j: {j}")
                    self.dist_matrix.append(int(round(float(num))))
    
    def find_min_dist(self, arr):
        low = 9999999
        low_i = -1
        unique = 0
        for i in range(self.N):
            if arr.count(i) == 0:
                md = self.find_min_dist(arr + [i]) + self._node_dist_manhattan(arr[-1], i) #np.sqrt(self._node_sqdist(arr[-1], i))
                # if md == 0:
                #     print(f'arr: {arr}, i: {i}, md: {md}')
                if md < low:
                    low = md
                    low_i = i
            else:
                unique += 1
        if unique == self.N:
            # print(f'min path: {arr}')
            # return np.sqrt(self._node_sqdist(arr[-1] ,arr[0]))
            return self._node_dist_manhattan(arr[-1], arr[0])
        return low
                
    def seed(self, seed):
        np.random.seed(seed)


def save_log(data, name):
    if len(data) == 0:
        print(f'{name} data len is 0')
        return
    avgs = []
    num_pts = 100
    for i in range(num_pts):
        sum = 0
        for i in range(len(data)//num_pts * i, len(data)//num_pts * (i+1)):
            sum += data[i]
        avgs.append(sum / (len(data) // num_pts))
    fig, ax = plt.subplots()
    ax.scatter(range(num_pts), avgs)
    fig.savefig(f'{data}_log.png')

def save_logs():
    save_log(steps_log, 'steps')
    save_log(dist_log, 'dist')
    save_log(solved_log, 'solved')
