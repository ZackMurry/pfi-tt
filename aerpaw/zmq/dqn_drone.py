from aerpawlib.runner import ZmqStateMachine, entrypoint, state, timed_state
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit

print('Starting...')

ZMQ_COORDINATOR = 'COORDINATOR'

class DQNDrone(ZmqStateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.actions = []
        self.action_idx = 0
        self.drone_idx = 0
        self.drone_dests = []
        self.dwt = True
        self.truck_location = 0
        self._takeoff_ordered = False
        self._next_step = False
        self._is_moving = False

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
        await drone.takeoff(60)
        print('Finished taking off! Reached 50m')
        self.start_pos = drone.position
        print(f"Start pos: {self.start_pos}")
        print("Sending callback to coordinator")
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_taken_off')
        print('Following route...')
        return "follow_route"

    @state(name="wait_for_step")
    async def wait_for_step(self, _):
        if self._next_step:
            return "follow_route"
        else:
            print('Waiting for step...')
            await asyncio.sleep(0.5)
            return "wait_for_step"
    
    @state(name="step")
    async def step(self, _):
        self._next_step = True
        return "wait_for_step"

    @state(name="follow_route")
    async def follow_route(self, drone: Drone):
        self._next_step = False
        if self.action_idx >= len(self.actions):
            return "land"

        will_wait = False
        action = self.actions[self.action_idx]
        if action == 0: # Drone operation
            if self.dwt: # If drone is with truck, go out to next drone dest
                action = self.drone_dests[self.drone_idx]
                self.drone_idx += 1
                # Truck should not wait on drone to deliver package before moving on
                print('Sending callback_drone_out')
                await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_out')
            else:
                action = self.truck_location
                # Drone needs to wait for the truck
                print('Will wait: true')
                will_wait = True
            self.dwt = not self.dwt
        else:
            self.truck_location = action
            if not self.dwt:
                self.action_idx += 1
                return "follow_route"
        cust = self.customers[action - 1]
        target_x = cust['x'] * 10
        target_y = cust['y'] * 10
        print(f"Next target: {target_x}, {target_y}")
        self.is_moving = True
        moving = asyncio.ensure_future(drone.goto_coordinates(self.start_pos + VectorNED(target_y, -target_x, 0)))
        self.action_idx += 1
        await moving
        self.is_moving = False

        if will_wait:
            print("Waiting for rover!")
            await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_wait_for_rover')
        elif self.dwt:
            print('Finished step [dwt]')
            await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_finished_step')
        else: # Not waiting and not with the truck = keep going (i.e., go to the future truck_location)
            return "follow_route"

        return "wait_for_step"

    @state(name="land")
    async def land(self, drone: Drone):
        print('Landing...')
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_drone_landed')
        await asyncio.ensure_future(drone.goto_coordinates(self.start_pos))
        await drone.land()
        print('Landed...')
