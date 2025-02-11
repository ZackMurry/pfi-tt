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
from DisruptedScenario import DisruptedScenario
from math import pow, sqrt

# todo: add requirement for truck and drone to end at depot
# todo: fix bug with some drone dests not showing routes
# todo: add dynamic by fixing some part of the route
class NetworkDisruptionEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.min_time = 999999
        self.DRONE_SPEED_FACTOR = 1
        self.SHOW_HEURISTIC = False
        self.scenario = DisruptedScenario()
        self.t = 0
        self.x = 0
        self.y = 0
        self.MAX_T = 151
        self.MAX_NODES = 25
        self.MAX_QUEUE = 10
        self.max_nodes_reached = self.MAX_QUEUE
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
        self.disruptions_enabled = True
        self.episodes = 0
        self.customers = [] # id, x, y, deadline
        self.proposed_route = []
        self.planned_route = []
        self.action_list = []
        self.depot = {
            "x": 0,
            "y": 0
        }
        self.drone_with_truck = True
        self.drone_route = []
        self.allow_drone = 0
        self.remaining = []

        self.proposed_time = 0
        self.rejections = 0
        self.max_rejections = 4
        self.draw_all = False
        self.total_time = 0
        self.step_limit = self.MAX_QUEUE + self.max_rejections


        # drone flag + truck path + empty
        max_queue_range = 1 + self.MAX_QUEUE + 1
        self.observation_space = spaces.Dict({
            "planned_route": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
            "route_length": spaces.Discrete(self.MAX_QUEUE+1),
            # Only contains drone's destinations -- timing is included in planned_route
            "drone_route": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
            "proposed_route": spaces.MultiDiscrete([max_queue_range]*self.MAX_QUEUE),
            "request": spaces.Dict({
                "x": spaces.Discrete(self.MAX_X),
                "y": spaces.Discrete(self.MAX_Y),
                "deadline": spaces.Discrete(self.MAX_T),
                "disrupted": spaces.Discrete(2)
            }),
            "customers": spaces.Dict({
                "x": spaces.MultiDiscrete([self.MAX_X]*self.MAX_QUEUE),
                "y": spaces.MultiDiscrete([self.MAX_Y]*self.MAX_QUEUE),
                "deadline": spaces.MultiDiscrete([self.MAX_T]*self.MAX_QUEUE),
                # "disrupted": spaces.MultiDiscrete([2]*self.MAX_QUEUE)
            }),
            "rejections": spaces.Discrete(self.max_rejections + 1)
        })

        # reject + drone flag + truck path
        self.action_space = spaces.Discrete(1+1+self.MAX_QUEUE, start=0)
        # self.action_space = spaces.Box(np.array([0,0]), np.array([2,3]), dtype=int)
        self.reset()
    


    def _STEP(self, action):
        done = False
        action -= 1 # Reject: -1, Drone: 0, Path: [1,n]
        if action-1 > len(self.planned_route):
            action = len(self.planned_route) + 1
        self.action_list.append(action)
        self.step_count += 1
        
        reward = 0
        ended = False
        # print(f"Action: {action}")
        if action == -1:
            self.rejections += 1
            self.rejected.append(self.request)
            self.request = None
            self.nodes_proposed -= 1
            self.step_count -= 1
            if self.rejections > self.max_rejections:
                print('Penalty: too many rejections')
                reward -= 4
                self.step_count += 1
                self.rejections = self.max_rejections
        elif self.request != None:
            if len(self.customers) < self.MAX_QUEUE:
                self.customers.append(self.request)
                if action == 0: # Send drone to capture request!
                    # print('drone')
                    if self.request['disrupted']:
                        # print(self.request)
                        reward -= 10
                        done = True
                        # done = True`
                    # else:
                    #     reward += 1
                    
                    if len(self.planned_route) > 1 and self.planned_route[-1] == 0: # Penalize making truck wait for drone to go out and come back
                        print(f'Penalizing drone action on {self.planned_route}')
                        reward -= 5 #3


                    if self.drone_with_truck:
                        # Send drone to next customer
                        self.planned_route.append(0)
                        # print('Appending to drone route', len(self.customers))
                        self.drone_route.append(len(self.customers))
                        self.drone_with_truck = False
                        reward += 1 #1 # 1 before, 2 after is best so far
                    else:
                        self.planned_route.append(0)
                        self.drone_with_truck = True
                        reward += 2 #2 # Give reward once truck collects drone (avoids ending without collecting)

                else:
                    # print('Adding to planned route', len(self.customers))
                    self.planned_route.insert(action, len(self.customers))
                    reward += 1
                self.request = None
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
        self.remaining = [-1] * len(self.customers)
        dwt = True # Drone with truck?
        drone_idx = 0
        drone_time = 0
        drone_start_time = 0
        for dest in self.planned_route:
            if dest == 0: # Drone delivery/pickup
                if dwt: # Send drone out
                    assert drone_idx < len(self.drone_route)
                    # print(f"Getting customer obj for cust {self.drone_route[drone_idx]} at drone_idx {drone_idx}")
                    cust = self.customers[self.drone_route[drone_idx]-1]
                    dt = self._get_travel_time(vx, vy, cust)
                    drone_time = dt / self.DRONE_SPEED_FACTOR
                    self.remaining[self.drone_route[drone_idx]-1] = cust['deadline'] - (time + drone_time)
                    # print('Remaining ', self.drone_route[drone_idx]-1, ' is ', cust['deadline'] - (time + drone_time))
                    drone_start_time = time
                    dwt = False
                    if time + drone_time > cust['deadline']:
                        reward -= 1
                        done = True
                        # break
                else: # Send drone back
                    cust = self.customers[self.drone_route[drone_idx]-1]
                    dt = self._get_travel_time(vx, vy, cust)
                    drone_time += dt / self.DRONE_SPEED_FACTOR
                    drone_idx += 1
                    time += max(drone_time - (time - drone_start_time), 0)
                    dwt = True
                continue


            cust = self.customers[dest-1]
            dt = self._get_travel_time(vx, vy, cust)
            if time + dt > cust['deadline']:
                done = True
                reward -= 1
                time += dt
                self.remaining[dest-1] = cust['deadline'] - time
            else:
                time += dt
                self.remaining[dest-1] = cust['deadline'] - time
                vx = cust.get('x')
                vy = cust.get('y')
        
        if time > 0 and dwt:
            if self.max_nodes_reached == len(self.planned_route):
                if self.min_time > time:
                    self.min_time = time
                    print(f"Min time for {self.max_nodes_reached}: {self.min_time}")
                    # self.draw_env(time, remaining)
            elif self.max_nodes_reached < len(self.planned_route):
                self.min_time = time
                self.max_nodes_reached = len(self.planned_route)
                print(f"Reached new node count! {self.max_nodes_reached}; time: {self.min_time}")

        self._update_state()
        if (done or self.step_count >= self.step_limit):
            num_drone_steps = 0
            for i in range(len(self.planned_route)):
                if self.planned_route[i] == 0:
                    num_drone_steps += 1
            if num_drone_steps % 2 == 1:
                self.planned_route.append(0)

        # if self.draw_all and (done or self.step_count >= self.step_limit):
        # print("Draw all?", self.draw_all)
        # if (self.episodes % 1000 == 0 or (self.episodes > 10000 and self.episodes % 250 == 0) or self.draw_all) and (done or self.step_count >= self.step_limit):
        self.total_time = time
        if (self.episodes % 1000 == 0 or self.draw_all) and (done or self.step_count >= self.step_limit):
            self.draw_env()
        
        # if self.step_count >= self.step_limit and not dwt:
            # reward -= 5

        if self.step_count >= self.step_limit and not done:
            reward += 3
        
        if self.step_count > 5 and 0 not in self.planned_route:
            reward -= 3

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
        self.total_time = 0
        self.x = 0
        self.y = 0
        self.visited = []
        self.remaining = []
        self.rejected = []
        self.nodes_proposed = 0
        self.proposed_time = 0
        self.step_count = 0
        self.rejections = 0
        self.customers = []
        self.scenario = DisruptedScenario()
        self.scenario.set_disruptions_enabled(self.disruptions_enabled)
        self.request = self.scenario.request()
        self.proposed_route = self._propose_route()
        self.planned_route = []
        self.action_list = []
        self.drone_with_truck = True
        self.drone_route = []
        self._update_state()
        return self.state

    def _update_state(self):
        self.proposed_route = self._propose_route()
        req = self.request
        if req == None:
            req = {
                "x": 0,
                "y": 0,
                "deadline": 0,
                "disrupted": 0
            }
        custs = {
            "x": [],
            "y": [],
            "deadline": [],
            # "disrupted": []
        }
        i = 1
        for cust in self.customers:
            custs['x'].append(cust['x'])
            custs['y'].append(cust['y'])
            custs['deadline'].append(cust['deadline'] - self.t)
            # custs['disrupted'].append(cust['disrupted'])
            i += 1
        
        while len(custs['x']) < self.MAX_QUEUE:
            custs['x'].append(0)
            custs['y'].append(0)
            custs['deadline'].append(0)
            # custs['disrupted'].append(0)

        proposed = self.proposed_route.copy()
        while len(proposed) < self.MAX_QUEUE:
            proposed.append(0)

        planned = self.planned_route.copy()
        while len(planned) < self.MAX_QUEUE:
            planned.append(0)

        drone_r = self.drone_route.copy()
        while len(drone_r) < self.MAX_QUEUE:
            drone_r.append(0)

        state = {
            "request": req,
            "customers": {
                "x": np.array(custs['x']),
                "y": np.array(custs['y']),
                "deadline": np.array(custs['deadline']),
                # "disrupted": np.array(custs['disrupted'])
            },
            "proposed_route": np.array(proposed),
            "planned_route": np.array(planned),
            "drone_route": np.array(drone_r),
            "route_length": len(self.planned_route),
            "rejections": self.rejections
        }
        # print({
        #     "request": req,
        #     "customers": {
        #         "x": len(np.array(custs['x'])),
        #         "y": len(np.array(custs['y'])),
        #         "deadline": len(np.array(custs['deadline'])),
        #         # "disrupted": np.array(custs['disrupted'])
        #     },
        #     "proposed_route": len(np.array(proposed)),
        #     "planned_route": len(np.array(planned)),
        #     "drone_route": len(np.array(drone_r)),
        #     "route_length": len(self.planned_route),
        #     "rejections": self.rejections
        # })
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

    def _get_travel_time_diag(self, x, y, customer):
        return sqrt(pow(customer.get('x') - x, 2) + pow(customer.get('y') - y, 2))

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

    def draw_env(self):
        fig, ax = plt.subplots(figsize=(12,8))
        # todo: export to file in standardized data format for AERPAW script

        with open(f"results/ep-{self.episodes}-n-{len(self.customers)}-t-{round(self.total_time, 2)}.plan", 'w') as f:
            f.write(f"{len(self.planned_route)}\n")
            for act in self.planned_route:
                f.write(f"{act} ")
            f.write('\n')
            f.write(f"{len(self.drone_route)}\n")
            for d_act in self.drone_route:
                f.write(f"{d_act} ")
            f.write('\n')


        ax.set_xlim([0,20])
        ax.set_ylim([0,20])
        ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        ax.set_yticks([0,2,4,6,8,10,12,14,16,18,20])
        ax.set_title(f'{self.action_list}')
        # route = self.visited + self.planned_route
        dwt = True
        drone_idx = 0
        node_idx = 1
        drawn = []
        for i in range(len(self.planned_route)):
            act = self.planned_route[i]
            if act == 0:
                if dwt:
                    if drone_idx >= len(self.drone_route):
                        print('drone idx too high')
                        print('drone', self.drone_route)
                        print('planned', self.planned_route)
                        return
                    cust_id = self.drone_route[drone_idx]-1
                    cust = self.customers[cust_id]
                    mark = '.'
                    if cust['disrupted'] == 1:
                        # print('drawing disrupted')
                        mark = 's'
                    if self.remaining[cust_id] < 0:
                        mark = 'X'
                    ax.scatter(cust['x'], cust['y'], color='tab:pink', s=200, marker=mark)
                    ax.annotate(f"${node_idx},d={cust['deadline']},t={round(cust['deadline'] - self.remaining[cust_id])}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                    node_idx += 1
                    drawn.append(cust_id)
                else:
                    drone_idx += 1 
                dwt = not dwt
            elif (act-1) not in drawn:
                cust = self.customers[act-1]
                mark = '.'
                if cust['disrupted'] == 1:
                    # print('drawing disrupted')
                    mark = 's'
                if self.remaining[act-1] < 0:
                    mark = 'X'
                ax.scatter(cust['x'], cust['y'], color='b', s=300, marker=mark)
                ax.annotate(f"${node_idx},d={cust['deadline']},t={round(cust['deadline'] - self.remaining[act-1])}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
                node_idx += 1
                drawn.append(act-1)



        # for i in range(len(self.customers)):
        #     cust = self.customers[i]
        #     col = 'g' if i == self.planned_route[-1]-1 else 'b'
        #     ax.scatter(cust['x'], cust['y'], color=col, s=300)
        #     ax.annotate(f"$N_{i},d={cust['deadline']},t={cust['deadline'] - remaining[i]}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
        for rej in self.rejected:
            ax.scatter(rej['x'], rej['y'], color='red', s=100)
        # for node in self.visited:
        #     col = 'black'
        #     ax.scatter(node['x'], node['y'], color=col, s=300)
        #     ax.annotate(f"$N_{i},d={node['deadline']}$", xy=(cust['x']+0.4, cust['y']+0.05), zorder=2)
        col = 'bo'
        for i in range(len(self.visited) - 1):
            ax.plot([self.customers[self.planned_route[i]-1]['x'], self.customers[self.planned_route[i+1]-1]['x']],
                [self.customers[self.planned_route[i]-1]['y'], self.customers[self.planned_route[i+1]-1]['y']], col, linestyle='solid')
        if self.total_time < 0:
            col = 'yo'
        if len(self.visited) > 0:
            ax.plot([0, self.visited[0]['x']],
                [0, self.visited[0]['y']], col, linestyle='solid')
            for i in range(len(self.visited)-1):
                ax.plot([self.visited[i]['x'], self.visited[i+1]['y']],
                    [self.visited[i+1]['x'], self.visited[i+1]['y']])
        
        # for dd in self.drone_route:
        #     cust = self.customers[dd-1]
        #     ax.scatter(cust['x'], cust['y'], color='tab:pink', s=200)


        truck_route = []
        for dest in self.planned_route:
            if dest != 0:
                truck_route.append(dest)
        if len(truck_route) > 0:
            ax.plot([self.x, self.customers[truck_route[0]-1]['x']],
                [self.y, self.customers[truck_route[0]-1]['y']], col, linestyle='solid')
        for i in range(len(truck_route) - 1):
            ax.plot([self.customers[truck_route[i]-1]['x'], self.customers[truck_route[i+1]-1]['x']],
                [self.customers[truck_route[i]-1]['y'], self.customers[truck_route[i+1]-1]['y']], col, linestyle='solid')
        if len(self.proposed_route) > 0 and self.SHOW_HEURISTIC:
            ax.plot([self.x, self.customers[self.proposed_route[0]-1]['x']],
                [self.y, self.customers[self.proposed_route[0]-1]['y']], 'ro', linestyle='--')
            for i in range(len(self.proposed_route) - 1):
                ax.plot([self.customers[self.proposed_route[i]-1]['x'], self.customers[self.proposed_route[i+1]-1]['x']],
                    [self.customers[self.proposed_route[i]-1]['y'], self.customers[self.proposed_route[i+1]-1]['y']], 'ro', linestyle='--')
        
        drone_idx = 0
        dwt = True # drone with truck
        vx = self.x # = 0
        vy = self.y # = 0
        for dest in self.planned_route:
            if dest == 0: # is drone
                if dwt:
                    ax.scatter(vx, vy, color='c', s=300)
                else:
                    ax.scatter(vx, vy, color='tab:brown', s=300)
                ax.plot([vx, self.customers[self.drone_route[drone_idx]-1]['x']],
                    [vy, self.customers[self.drone_route[drone_idx]-1]['y']],
                    'mo', linestyle='dashed')
                
                if not dwt:
                    drone_idx += 1

                dwt = not dwt
            else:
                cust = self.customers[dest-1]
                vx = cust.get('x')
                vy = cust.get('y')




        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.savefig(f"results/ep-{self.episodes}-n-{len(self.customers)}-t-{round(self.total_time, 2)}.png")
    
    def set_draw_all(self, val):
        self.draw_all = val
    
    def set_disruptions_enabled(self, val):
        self.disruptions_enabled = val
        self.scenario.set_disruptions_enabled(val)



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
