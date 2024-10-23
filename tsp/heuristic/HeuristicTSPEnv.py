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

perfect_log = []

class HeuristicTSPEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.min_time = 999999
        self.max_nodes_reached = 0
        self.scenario = TSPScenario()
        self.t = 0
        self.x = 0
        self.y = 0
        self.MAX_T = 116
        self.MAX_NODES = 25
        self.MAX_QUEUE = 15
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
        # customers is sorted in order of ID
        self.customers = [
            # {
            #     "id": 0,
            #     "x": 3,
            #     "y": 5,
            #     "deadline": 10
            # },
            # {
            #     "id": 1,
            #     "x": 2,
            #     "y": 1,
            #     "deadline": 15
            # },
            # {
            #     "id": 2,
            #     "x": 7,
            #     "y": 10,
            #     "deadline": 30
            # },
            # {
            #     "id": 3,
            #     "x": 5,
            #     "y": 8,
            #     "deadline": 30
            # },
        ] # id, x, y, deadline
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
                # "id": spaces.Discrete(max_queue_range),
                "x": spaces.Discrete(self.MAX_X),
                "y": spaces.Discrete(self.MAX_Y),
                "deadline": spaces.Discrete(self.MAX_T)
            }),
            "customers": spaces.Dict({
                # "id": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
                "x": spaces.MultiDiscrete([self.MAX_X]*self.MAX_QUEUE),
                "y": spaces.MultiDiscrete([self.MAX_Y]*self.MAX_QUEUE),
                "deadline": spaces.MultiDiscrete([self.MAX_T]*self.MAX_QUEUE)
            }),
            "rejections": spaces.Discrete(self.max_rejections + 1),
        })

        self.action_space = spaces.Discrete(self.MAX_QUEUE+1, start=0)
        # self.action_space = spaces.Box(np.array([0,0]), np.array([2,3]), dtype=int)
        self.is_perfect = True
        self.reset()
    


    def _STEP(self, action):
        done = False
        if action-1 > len(self.planned_route):
            action = len(self.planned_route) + 1
        self.action_list.append(action)
        self.step_count += 1
        
        # route = get_route_from_action(119)
        # print(f"route: {route}")


        reward = 0
        # print(f"Action: {action}")
        if action == 0:
            self.rejections += 1
            self.rejected.append(self.request)
            # if self.rejections > self.max_rejections:
            #     print('Too many rejections!')
            #     reward -= 4
            # reward -= 1
            self.request = None
            self.nodes_proposed -= 1
        elif self.request != None:
            # if action-1 > len(self.planned_route):
            #     action = len(self.planned_route) + 1
            if len(self.customers) < self.MAX_QUEUE:
                self.customers.append(self.request)
                self.planned_route.insert(action-1, len(self.customers))
                self.request = None
                reward += 1
            else:
                done = True
        else:
            reward -= 1 # No request made
            self.is_perfect = False
            print(f'Error: no request made; {len(self.customers)}')
        
        if self.request == None:
            self.request = self.scenario.request()

        # if len(self.customers) >= self.MAX_QUEUE:
        #     reward += 3
        #     done = True


        # print(f"Action: {action}; route: {self.planned_route}")
        # Validate path
        time = self.t
        vx = self.x
        vy = self.y
        remaining = [-1] * len(self.customers)
        for dest in self.planned_route:
            cust = self.customers[dest-1]
            dt = self._get_travel_time(vx, vy, cust)
            if time + dt > cust['deadline']:
                # print(f'Missed deadline! {time + dt} vs. {cust["deadline"]}')
                # print(f'Error: missed deadline of {cust["deadline"]}')
                # self.is_perfect = False
                done = True
                # if not done:
                #     reward -= 3
                #     perfect_log.append(0)
                #     done = True
                time += dt
                remaining[dest-1] = cust['deadline'] - time
                # time = -1
                # to_remove.append(i)
                # break
            else:
                time += dt
                # remaining.append(cust['deadline']-time)
                remaining[dest-1] = cust['deadline'] - time
                vx = cust.get('x')
                vy = cust.get('y')

        if time > 0:
            if self.max_nodes_reached == len(self.planned_route):
                if self.min_time > time:
                    # reward += 1
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
                        # ax.annotate(f"$N_{i},d={cust['deadline']},t={cust['deadline'] - remaining[i]},dt={remaining[i]}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                        ax.annotate(f"$N_{i},d={cust['deadline']},t={cust['deadline'] - remaining[i]}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    for rej in self.rejected:
                        ax.scatter(rej['x'], rej['y'], color='red', s=100)
                    for node in self.visited:
                        col = 'black'
                        ax.scatter(node['x'], node['y'], color=col, s=300)
                        ax.annotate(f"$N_{i},d={node['deadline']}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    col = 'go' if time <= self.proposed_time else 'bo'
                    for i in range(len(self.visited) - 1):
                        # print(f"i: {i}, custs: {len(self.customers)}, planned_route: {self.planned_route}, planned_route[i]: {self.planned_route[i]}")
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
                        # print(f"i: {i}, custs: {len(self.customers)}, planned_route: {self.planned_route}, planned_route[i]: {self.planned_route[i]}")
                        ax.plot([self.customers[self.planned_route[i]-1]['x'], self.customers[self.planned_route[i+1]-1]['x']],
                            [self.customers[self.planned_route[i]-1]['y'], self.customers[self.planned_route[i+1]-1]['y']], col, linestyle='solid')
                    ax.plot([self.x, self.customers[self.proposed_route[0]-1]['x']],
                        [self.y, self.customers[self.proposed_route[0]-1]['y']], 'ro', linestyle='--')
                    for i in range(len(self.proposed_route) - 1):
                        ax.plot([self.customers[self.proposed_route[i]-1]['x'], self.customers[self.proposed_route[i+1]-1]['x']],
                            [self.customers[self.proposed_route[i]-1]['y'], self.customers[self.proposed_route[i+1]-1]['y']], 'ro', linestyle='--')
                    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fig.savefig(f"results/n-{self.max_nodes_reached}-t-{time}.png")
                    # self.min_time = min(self.min_time, time)
            elif self.max_nodes_reached < len(self.planned_route):
                self.min_time = time
                self.max_nodes_reached = len(self.planned_route)
                print(f"Reached new node count! {self.max_nodes_reached}; time: {self.min_time}")

        # if len(self.planned_route) == self.MAX_QUEUE and not done:
        #     first = self.customers.pop(self.planned_route.pop(0)-1)
        #     self.t += self._get_travel_time(self.x, self.y, first)
        #     self.x = first.get('x')
        #     self.y = first.get('y')
        #     self.visited.append(first)
        #     for i in range(len(self.planned_route)):
        #         self.planned_route[i] -= 1
        
        # if len(self.visited) >= self.MAX_NODES:
        #     reward += 3
        #     done = True

        if done and False:
            perfect_log.append(1 if self.is_perfect else 0)
            # print(f"route: {self.planned_route}, customers: {self.customers}")
            self.proposed_route = self._propose_route()
            # print(f"Time: {time}; custs: {self.customers}")
            # if time > 0 and len(self.customers) >= self.MAX_QUEUE:
                # print(f"Time: {time}")
                # reward -= time / 10
            # if time > self.proposed_time:
            #     reward -= 1
            # elif time > 0:
            #     print('better than proposed!')
            #     reward += 1.5 * (self.proposed_time / time)
            # if self.is_perfect:
            #     print('perfect!')
                # reward += 3
            # todo: show real arrival time
            #print("episodes: " + self.episodes)
            if self.episodes % 1000 == 0:
                print(f"Proposed route: {self.proposed_route}")
                print(f"Generated route: {self.planned_route}")
                print(f"Time {time} vs. Proposed Time {self.proposed_time}")
                if False:
                    fig, ax = plt.subplots(figsize=(12,8))
                    ax.set_xlim([0,20])
                    ax.set_ylim([0,20])
                    ax.set_title(f'{self.action_list}')
                    route = self.visited + self.planned_route
                    for i in range(len(self.customers)):
                        cust = self.customers[i]
                        col = 'g' if i == self.planned_route[-1]-1 else 'b'
                        ax.scatter(cust['x'], cust['y'], color=col, s=300)
                        ax.annotate(f"$N_{i},d={cust['deadline']},t={cust['deadline'] - remaining[i]},dt={remaining[i]}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    for node in self.visited:
                        col = 'black'
                        ax.scatter(node['x'], node['y'], color=col, s=300)
                        ax.annotate(f"$N_{i},d={node['deadline']}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    col = 'go' if time <= self.proposed_time else 'bo'
                    for i in range(len(self.visited) - 1):
                        # print(f"i: {i}, custs: {len(self.customers)}, planned_route: {self.planned_route}, planned_route[i]: {self.planned_route[i]}")
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
                        # print(f"i: {i}, custs: {len(self.customers)}, planned_route: {self.planned_route}, planned_route[i]: {self.planned_route[i]}")
                        ax.plot([self.customers[self.planned_route[i]-1]['x'], self.customers[self.planned_route[i+1]-1]['x']],
                            [self.customers[self.planned_route[i]-1]['y'], self.customers[self.planned_route[i+1]-1]['y']], col, linestyle='solid')
                    ax.plot([self.x, self.customers[self.proposed_route[0]-1]['x']],
                        [self.y, self.customers[self.proposed_route[0]-1]['y']], 'ro', linestyle='--')
                    for i in range(len(self.proposed_route) - 1):
                        ax.plot([self.customers[self.proposed_route[i]-1]['x'], self.customers[self.proposed_route[i+1]-1]['x']],
                            [self.customers[self.proposed_route[i]-1]['y'], self.customers[self.proposed_route[i+1]-1]['y']], 'ro', linestyle='--')
                    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fig.savefig(f"results/{self.MAX_QUEUE}-solution-{current_datetime}.png")
        
        self._update_state()
        
        if self.step_count >= self.step_limit and not done:
            perfect_log.append(0)
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
        if len(perfect_log) > 100 and len(perfect_log) % 100 == 0:
            perf = 0
            for i in range(len(perfect_log)-100, len(perfect_log)):
                if perfect_log[i] == 1:
                    perf += 1
            # print(f'{perf}% perfect')
        # print("RESET------------------------------------------")
        self.is_perfect = True
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
        # self.request = None
        self.scenario = TSPScenario()
        self.request = self.scenario.request()
        # self.generate_1d_distance_matrix()
        self.proposed_route = self._propose_route()
        self.planned_route = []
        self.action_list = []
        self._update_state()
        return self.state
        
    def _generate_request(self):
        self.nodes_proposed += 1
        min_bound = 30 + (self.nodes_proposed-1)*10
        min_bound = max(min_bound, 40)
        return {
            "x": np.random.randint(0, self.MAX_X),
            "y": np.random.randint(0, self.MAX_Y),
            "deadline": np.random.randint(min_bound, max(min_bound, 35 + self.nodes_proposed*10))
        }
        

    def _update_state(self):
        self.proposed_route = self._propose_route()
        req = self.request
        if req == None:
            req = {
                # "id": 0, # id of 0 => null
                "x": 0,
                "y": 0,
                "deadline": 0
            }
        # else:
        #     req = {
        #         "x": req['x'],
        #         "y": req['y']
        #     }
        custs = {
            # "id": [],
            "x": [],
            "y": [],
            "deadline": []
        }
        i = 1
        for cust in self.customers:
            # custs['id'].append(i)
            custs['x'].append(cust['x'])
            custs['y'].append(cust['y'])
            custs['deadline'].append(cust['deadline'] - self.t)
            i += 1
        
        while len(custs['x']) < self.MAX_QUEUE:
            # custs['id'].append(0) # id of 0 => null
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
    fig.savefig(f'{name}_log.png')

def save_logs():
    # save_log(steps_log, 'steps')
    # save_log(dist_log, 'dist')
    # save_log(solved_log, 'solved')
    save_log(perfect_log, 'perfect')
