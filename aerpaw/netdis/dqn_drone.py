from aerpawlib.runner import ZmqStateMachine, entrypoint, state, timed_state, expose_field_zmq
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit
import random
import torch
from torch import nn
import numpy as np
from HeuristicTruckDroneEnv import HeuristicTruckDroneEnv
from gymnasium.wrappers import FlattenObservation
from interface import get_path

class Net(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True), # new
            # nn.Linear(128, 128), nn.ReLU(inplace=True), # new
            # nn.Linear(128, 128), nn.ReLU(inplace=True), # new
            nn.Linear(128, np.prod(action_shape))#, device=device),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)#, device=device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        # print(f"logits: {logits}")
        return logits, state

print('Starting...')

ZMQ_COORDINATOR = 'COORDINATOR'

env = FlattenObservation(HeuristicTruckDroneEnv())
state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")

model = Net(state_shape, action_shape)
model.load_state_dict(torch.load("netdis_policy.pth"))


mock = False

GRID_SIZE = 15

class DQNDrone(ZmqStateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.dwt = True
        self._takeoff_ordered = False
        self._next_step = False
        self._is_moving = False
        self.network_disrupted = False
        self.finished = True
        self.sleeps = 0
        
        scenario_file_name = '/root/netdis.scenario'
        print(f"Reading scenario file from {scenario_file_name}...")
        self.customers = []
        with open(scenario_file_name, 'r') as file:
            n = int(file.readline())
            print(f"Reading {n} customers")
            for i in range(n):
                details = file.readline().split(' ')
                if len(details) != 3:
                    print(f"Error: expected 3 details but got: {details}; aborting...")
                    exit(0)
                cust = {
                    'x': int(details[0]),
                    'y': int(details[1]),
                    'deadline': int(details[2])
                }
                self.customers.append(cust)

        print('Finished reading data!')
        
    @state(name="wait_for_start", first=True)
    async def wait_for_start(self, _):
        if self._takeoff_ordered:
            return "start"
        else:
            print('Waiting for start...')
            await asyncio.sleep(0.1)
            return "wait_for_start"

    @state(name="take_off")
    async def take_off(self, _):
        print('Takeoff ordered!')
        self._takeoff_ordered = True
        return "wait_for_start"

    @state(name="start")
    async def start(self, drone: Drone):
        print('Taking off...')
        if not mock:
            await drone.takeoff(60)
        print('Finished taking off! Reached 50m')
        self.start_pos = drone.position
        print(f"Start pos: {self.start_pos}")
        print("Sending callback to coordinator")
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_taken_off')
        print('Following route...')
        return "follow_route"

    @expose_field_zmq(name="ping")
    async def ping(self, _):
        if self.network_disrupted:
            print("not responding to ping")
            await asyncio.sleep(5)
            return None
        else:
            return "pong"

    @state(name="wait_for_step")
    async def wait_for_step(self, _):
        if self._next_step:
            self.sleeps = 0
            return "follow_route"
        else:
            print('Waiting for step...')
            await asyncio.sleep(0.5)
            self.sleeps += 1
            if self.sleeps > 30:
                print('re-sending callback_drone_finished_step')
                await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_finished_step')
                self.sleeps = 0
            return "wait_for_step"
    
    @state(name="step")
    async def step(self, _):
        self._next_step = True
        self.finished = False
        return "wait_for_step"

    @state(name="follow_route")
    async def follow_route(self, drone: Drone):
        print('following route!')
        self._next_step = False
        next_waypoint = await self.query_field(ZMQ_COORDINATOR, "drone_next")
        print("next_waypoint: ", next_waypoint)

        if next_waypoint == -1:
            return "land"

        start = drone.position  # Must be lat/lon
        cust = self.customers[next_waypoint - 1]
        goal = self.start_pos + VectorNED(cust['y'] * 10, -cust['x'] * 10, 0)

        # We force the RL path to operate on a fixed grid space
        grid_start = (0, 0)
        grid_goal = (GRID_SIZE - 1, GRID_SIZE - 1)

        def grid_to_latlon(gx, gy):
            frac_x = gx / (GRID_SIZE - 1)
            frac_y = gy / (GRID_SIZE - 1)

            lat = start.lat + frac_y * (goal.lat - start.lat)
            lon = start.lon + frac_x * (goal.lon - start.lon)
            return Coordinate(lat=lat, lon=lon, alt=start.alt)

        obstacles = []  # still empty unless you want to map no-fly zones etc.

        path = get_path(grid_start, grid_goal, obstacles)
        print(path)

        num_waypoints = 5
        if len(path) <= num_waypoints:
            sampled_points = path
        else:
            indices = [round(i * (len(path) - 1) / (num_waypoints - 1)) for i in range(num_waypoints)]
            sampled_points = [path[i] for i in indices]
        
        print(sampled_points)

        self.is_moving = True
        for (gx, gy) in sampled_points[1:]:  # skip first (it's current position)
            target = grid_to_latlon(gx, gy)
            print(f"Moving to ({gx},{gy}) => {target}")
            await drone.goto_coordinates(target)
        self.is_moving = False
        self.finished = True
        print('sending callback_drone_finished_step')
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_finished_step')

        return "wait_for_step"

    # @state(name="follow_route")
    # async def follow_route(self, drone: Drone):
    #     print('following route!')
    #     self._next_step = False
    #     next_waypoint = await self.query_field(ZMQ_COORDINATOR, "drone_next")
    #     print("next_waypoint: ", next_waypoint)

        
    #     if next_waypoint == -1:
    #         return "land"

    #     start = drone.position
    #     start.lon = -1 * start.lon
    #     cust = self.customers[next_waypoint - 1]

    #     goal = self.start_pos + VectorNED(cust['y'] * 20, cust['x'] * 20, 0)
    #     goal.lon = -1 * goal.lon


    #     # Normalize to (0, 0) – (14, 14)
    #     # Create bounds of the rectangle
    #     min_lat = min(start.lat, goal.lat)
    #     max_lat = max(start.lat, goal.lat)
    #     min_lon = min(start.lon, goal.lon)
    #     max_lon = max(start.lon, goal.lon)

    #     # Mapping functions
    #     def latlon_to_grid(lat, lon):
    #         gx = int(round((lon - min_lon) / (max_lon - min_lon) * (GRID_SIZE - 1)))
    #         gy = int(round((lat - min_lat) / (max_lat - min_lat) * (GRID_SIZE - 1)))
    #         return (gy, gx)

    #     def grid_to_latlon(gx, gy):
    #         lon = min_lon + (gx / (GRID_SIZE - 1)) * (max_lon - min_lon)
    #         lat = min_lat + (gy / (GRID_SIZE - 1)) * (max_lat - min_lat)
    #         return Coordinate(lat=lat, lon=lon, alt=start.alt)

    #     grid_start = latlon_to_grid(start.lat, start.lon)
    #     grid_goal = latlon_to_grid(goal.lat, goal.lon)

    #     print(f"Mapped start {start} to grid {grid_start}")
    #     print(f"Mapped goal {goal} to grid {grid_goal}")

    #     # Obstacles — empty or generated based on domain knowledge
    #     obstacles = []  # or generate inside this rectangle if needed

    #     # Get RL path
    #     path = get_path(grid_start, grid_goal, obstacles)
    #     # print("RL path:", path)
        
    #     # Follow a simplified path with 5 waypoints (including end)
    #     self.is_moving = True
    #     num_waypoints = 5
    #     path_len = len(path)

    #     if path_len <= num_waypoints:
    #         sampled_points = path
    #     else:
    #         # Pick 5 evenly spaced points including start and end
    #         indices = [round(i * (path_len - 1) / (num_waypoints - 1)) for i in range(num_waypoints)]
    #         sampled_points = [path[i] for i in indices]

    #     for (gy, gx) in sampled_points[1:]:  # skip current position
    #         target_ned = grid_to_latlon(gx, gy)
    #         target_ned.lon = -1 * target_ned.lon
    #         print(f"Moving to grid ({gx},{gy}) -> NED {target_ned}")
    #         await drone.goto_coordinates(target_ned)
    #     self.is_moving = False



        # target_x = cust['x'] * 10
        # target_y = cust['y'] * 10
        # print(f"Next target: {target_x}, {target_y}")
        # self.is_moving = True
        # target_gps = self.start_pos + VectorNED(target_y, -target_x, 0)

        # if not mock:
        #     moving = asyncio.ensure_future(drone.goto_coordinates(self.start_pos + VectorNED(target_y, -target_x, 0)))
        #     await moving
        # else:
        #     print('sleeping!')
        #     await asyncio.sleep(random.uniform(0.5,5))
        #     print('done sleeping!')
        # self.is_moving = False

        # if next_waypoint == 5: # Hardcoded for now
        #     print('sending disruption!')
        #     await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_disrupted')
        #     self.is_moving = True
        #     print('moving from ', drone.position)
        #     print('moving to ', step_start_pos)
        #     if not mock:
        #         moving = asyncio.ensure_future(drone.goto_coordinates(step_start_pos)) # Go back to start point for step
        #         await moving
        #     else:
        #         print('sleeping!')
        #         await asyncio.sleep(random.uniform(0.5,5))
        #         print('done sleeping!')
        #     print('new pos ', drone.position)
        #     self.is_moving = False
        #     self.finished = True
        #     await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_finished_step')
        # else:

    @state(name="land")
    async def land(self, drone: Drone):
        print('Landing...')
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_landed')
        if not mock:
            await asyncio.ensure_future(drone.goto_coordinates(self.start_pos))
            await drone.land()
        print('Landed...')
    

    @expose_field_zmq(name="finished")
    async def is_drone_finished(self, _):
        return self.finished
