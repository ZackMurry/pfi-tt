from aerpawlib.runner import StateMachine, entrypoint, state, timed_state
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit

print('Starting...')

# todo: convert to a drone and do everything 60 ft in the air
# todo: aerpawlib zmq for coordinating actions (i.e., wait to meet truck before moving on)

class DQNRover(StateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.actions = []
        self.action_idx = 0
        self.drone_dests = []

        plan_file_name = '/root/n-15-t-88.0.plan'
        print(f"Reading plan from {plan_file_name}...")
        with open(plan_file_name, 'r') as file:
            n = int(file.readline()) # num actions
            self.actions = list(map(int, file.readline().split()))
            print(f"actions: {self.actions}")
            m = int(file.readline()) # num drone dests
            self.drone_dests = list(map(int, file.readline().split()))
            print(f"drone_dests: {self.drone_dests}")
        
        scenario_file_name = '/root/export_25.scenario'
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
        
    @state(name="start", first=True)
    async def start(self, rover: Drone):
        print('Starting rover...')
        await rover.takeoff(50)
        self.start_pos = rover.position
        print(f"Start pos: {self.start_pos}")
        print('Following route...')
        return "follow_route"

    @state(name="follow_route")
    async def follow_route(self, rover: Drone):
        if self.action_idx >= len(self.actions):
            return "park"

        action = self.actions[self.action_idx]
        if action == 0: # Drone operation
            self.action_idx += 1
            return "follow_route"
        cust = self.customers[action - 1]
        target_x = cust['x'] * 10
        target_y = cust['y'] * 10
        print(f"Next target: {target_x}, {target_y}")
        moving = asyncio.ensure_future(rover.goto_coordinates(self.start_pos + VectorNED(target_y, -target_x, 0)))
        self.action_idx += 1

        await moving

        return "follow_route"

    @state(name="park")
    async def park(self, rover: Drone):
        print('Parking...')
        await asyncio.ensure_future(rover.goto_coordinates(self.start_pos))
        await rover.land()
        print('Parked!')

