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
from TSPScenario import TSPScenario

class HeuristicTSPEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.min_time = 999999
        self.max_nodes_reached = 0
        self.scenario = TSPScenario()
        self.t = 0
        self.x = 0
        self.y = 0
        self.MAX_T = 156
        self.MAX_NODES = 35
        self.MAX_QUEUE = 25
        self.nodes_proposed = 0
        self.use_dataset = False
        self.step_count = 0
        self.visited = []
        self.rejected = []
        self.spec = SimpleNamespace(reward_threshold=100)
        self.render_ready = False
        self.MAX_X = 20
        self.MAX_Y = 20
        self.request = None
        self.generated_map = False
        self.episodes = 0
        self.customers = [] # id, x, y, deadline
        self.proposed_route = []
        self.planned_route = []
        self.action_list = []
        self.depot = {
            "x": 0,
            "y": 0
        }

        self.step_limit = self.MAX_QUEUE + 2
        self.proposed_time = 0
        self.rejections = 0
        self.max_rejections = self.MAX_QUEUE//2

        max_queue_range = self.MAX_QUEUE + 1
        self.observation_space = spaces.Dict({
            "planned_route": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
            "route_length": spaces.Discrete(self.MAX_QUEUE+1),
            "proposed_route": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
            "request": spaces.Dict({
                "x": spaces.Discrete(self.MAX_X),
                "y": spaces.Discrete(self.MAX_Y),
                "deadline": spaces.Discrete(self.MAX_T)
            }),
            "customers": spaces.Dict({
                "x": spaces.MultiDiscrete([self.MAX_X]*self.MAX_QUEUE),
                "y": spaces.MultiDiscrete([self.MAX_Y]*self.MAX_QUEUE),
                "deadline": spaces.MultiDiscrete([self.MAX_T]*self.MAX_QUEUE)
            }),
            "rejections": spaces.Discrete(self.max_rejections + 1),
        })

        self.action_space = spaces.Discrete(self.MAX_QUEUE+1, start=0)
        # self.action_space = spaces.Box(np.array([0,0]), np.array([2,3]), dtype=int)
        self.reset()
    


    def _STEP(self, action):
        done = False
        if action-1 > len(self.planned_route):
            action = len(self.planned_route) + 1
        self.action_list.append(action)
        self.step_count += 1
        
        reward = 0
        # print(f"Action: {action}")
        if action == 0:
            self.rejections += 1
            self.rejected.append(self.request)
            self.request = None
            self.nodes_proposed -= 1
        elif self.request != None:
            if len(self.customers) < self.MAX_QUEUE:
                self.customers.append(self.request)
                self.planned_route.insert(action-1, len(self.customers))
                self.request = None
                reward += 1
            else:
                done = True
        else:
            reward -= 1 # No request made
            print(f'Error: no request made; {len(self.customers)}')
        
        if self.request == None:
            self.request = self.scenario.request()

        # Validate path
        time = self.t
        vx = self.x
        vy = self.y
        remaining = [-1] * len(self.customers)
        for dest in self.planned_route:
            cust = self.customers[dest-1]
            dt = self._get_travel_time(vx, vy, cust)
            if time + dt > cust['deadline']:
                done = True
                time += dt
                remaining[dest-1] = cust['deadline'] - time
            else:
                time += dt
                remaining[dest-1] = cust['deadline'] - time
                vx = cust.get('x')
                vy = cust.get('y')

        if time > 0:
            if self.max_nodes_reached == len(self.planned_route):
                if self.min_time > time:
                    self.min_time = time
                    print(f"Min time for {self.max_nodes_reached}: {self.min_time}")
                    fig, ax = plt.subplots(figsize=(12,8))
                    ax.set_xlim([0,20])
                    ax.set_ylim([0,20])
                    ax.set_title(f'{self.action_list}')
                    route = self.visited + self.planned_route
                    for i in range(len(self.customers)):
                        cust = self.customers[i]
                        col = 'g' if i == self.planned_route[-1]-1 else 'b'
                        ax.scatter(cust['x'], cust['y'], color=col, s=300)
                        ax.annotate(f"$N_{i},d={cust['deadline']},t={cust['deadline'] - remaining[i]}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    for rej in self.rejected:
                        ax.scatter(rej['x'], rej['y'], color='red', s=100)
                    for node in self.visited:
                        col = 'black'
                        ax.scatter(node['x'], node['y'], color=col, s=300)
                        ax.annotate(f"$N_{i},d={node['deadline']}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    col = 'go' if time <= self.proposed_time else 'bo'
                    for i in range(len(self.visited) - 1):
                        ax.plot([self.customers[self.planned_route[i]-1]['x'], self.customers[self.planned_route[i+1]-1]['x']],
                            [self.customers[self.planned_route[i]-1]['y'], self.customers[self.planned_route[i+1]-1]['y']], col, linestyle='solid')
                    if time < 0:
                        col = 'yo'
                    if len(self.visited) > 0:
                        ax.plot([0, self.visited[0]['x']],
                            [0, self.visited[0]['y']], col, linestyle='solid')
                        for i in range(len(self.visited)-1):
                            ax.plot([self.visited[i]['x'], self.visited[i+1]['y']],
                                [self.visited[i+1]['x'], self.visited[i+1]['y']])


                    ax.plot([self.x, self.customers[self.planned_route[0]-1]['x']],
                        [self.y, self.customers[self.planned_route[0]-1]['y']], col, linestyle='solid')
                    for i in range(len(self.planned_route) - 1):
                        ax.plot([self.customers[self.planned_route[i]-1]['x'], self.customers[self.planned_route[i+1]-1]['x']],
                            [self.customers[self.planned_route[i]-1]['y'], self.customers[self.planned_route[i+1]-1]['y']], col, linestyle='solid')
                    ax.plot([self.x, self.customers[self.proposed_route[0]-1]['x']],
                        [self.y, self.customers[self.proposed_route[0]-1]['y']], 'ro', linestyle='--')
                    for i in range(len(self.proposed_route) - 1):
                        ax.plot([self.customers[self.proposed_route[i]-1]['x'], self.customers[self.proposed_route[i+1]-1]['x']],
                            [self.customers[self.proposed_route[i]-1]['y'], self.customers[self.proposed_route[i+1]-1]['y']], 'ro', linestyle='--')
                    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fig.savefig(f"results/n-{self.max_nodes_reached}-t-{time}.png")
            elif self.max_nodes_reached < len(self.planned_route):
                self.min_time = time
                self.max_nodes_reached = len(self.planned_route)
                print(f"Reached new node count! {self.max_nodes_reached}; time: {self.min_time}")

        self._update_state()
        
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
        self.episodes += 1
        self.t = 0
        self.x = 0
        self.y = 0
        self.visited = []
        self.rejected = []
        self.nodes_proposed = 0
        self.proposed_time = 0
        self.step_count = 0
        self.rejections = 0
        self.customers = []
        self.scenario = TSPScenario()
        self.request = self.scenario.request()
        self.proposed_route = self._propose_route()
        self.planned_route = []
        self.action_list = []
        self._update_state()
        return self.state

    def _update_state(self):
        self.proposed_route = self._propose_route()
        req = self.request
        if req == None:
            req = {
                "x": 0,
                "y": 0,
                "deadline": 0
            }
        custs = {
            "x": [],
            "y": [],
            "deadline": []
        }
        i = 1
        for cust in self.customers:
            custs['x'].append(cust['x'])
            custs['y'].append(cust['y'])
            custs['deadline'].append(cust['deadline'] - self.t)
            i += 1
        
        while len(custs['x']) < self.MAX_QUEUE:
            custs['x'].append(0)
            custs['y'].append(0)
            custs['deadline'].append(0)

        proposed = self.proposed_route.copy()
        while len(proposed) < self.MAX_QUEUE:
            proposed.append(0)

        planned = self.planned_route.copy()
        while len(planned) < self.MAX_QUEUE:
            planned.append(0)

        state = {
            "request": req,
            "customers": {
                "x": np.array(custs['x']),
                "y": np.array(custs['y']),
                "deadline": np.array(custs['deadline']),
            },
            "proposed_route": np.array(proposed),
            "planned_route": np.array(planned),
            "route_length": len(self.planned_route),
            "rejections": min(self.rejections, 2)
        }
        # print(f'state: {state}')
        self.state = state
        return state
    
    def _propose_route(self):
        x = self.depot.get('x')
        y = self.depot.get('y')
        visited = []
        to_visit = []
        for i in range(len(self.customers)):
            to_visit.append({
                "id": i,
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
            # print(f"Min t: {min_t}s from ({x}, {y}) to Customer {min_t_id} at ({cx}, {cy})")
            x = cx
            y = cy
            time += min_t
            visited.append(min_t_id+1)
            to_visit.pop(min_t_i)

        # print(f"Total time: {time}")
        self.proposed_time = time
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
    fig.savefig(f'{name}_log.png')

def save_logs():
    print("Saving logs...")
    # save_log(steps_log, 'steps')
    # save_log(dist_log, 'dist')
    # save_log(solved_log, 'solved')
