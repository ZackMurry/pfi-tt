from aerpawlib.runner import ZmqStateMachine, entrypoint, state, timed_state
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit

print('Starting...')

# todo: convert to a drone and do everything 60 ft in the air
# todo: aerpawlib zmq for coordinating actions (i.e., wait to meet truck before moving on)

ZMQ_COORDINATOR = 'COORDINATOR'

class DQNRover(ZmqStateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.takeoff_pos = None
        self.actions = []
        self.action_idx = 0
        self.drone_dests = []
        self._takeoff_ordered = False
        self._next_step = False
        self._dwt = True

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
            await asyncio.sleep(0.5)
            return "wait_for_start"
        
    @state(name="take_off")
    async def take_off(self, _):
        print('Takeoff ordered!')
        self._takeoff_ordered = True
        return "wait_for_start"

    @state(name="start")
    async def start(self, rover: Drone):
        print('Received takeoff message!')
        print('Starting rover...')
        await rover.takeoff(50)
        self.takeoff_pos = rover.position
        print('Moving under drone...')
        await asyncio.ensure_future(rover.goto_coordinates(Coordinate(35.7274825,-78.696275,50)))
        self.start_pos = rover.position
        print(f"Start pos: {self.start_pos}")
        print('Sending callback to coordinator...')
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_taken_off')
        print('Following route...')
        return "wait_for_step"

    @state(name="wait_for_step")
    async def wait_for_step(self, _):
        if self._next_step:
            print('Received STEP!')
            return "follow_route"
        else:
            print('Waiting for step...')
            await asyncio.sleep(0.1)
            return "wait_for_step"
    
    @state(name="step")
    async def step(self, _):
        self._next_step = True
        return "wait_for_step"

    @state(name="follow_route")
    async def follow_route(self, rover: Drone):
        self._next_step = False
        print(f"Action idx: {self.action_idx}")
        if self.action_idx >= len(self.actions):
            return "park"

        # await self.transition_runner('drone', 'test')
        action = self.actions[self.action_idx]
        if action == 0: # Drone operation
            self.action_idx += 1
            self._dwt = not self._dwt
            if self._dwt:
                print('Waiting for drone!')
                await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_wait_for_drone')
                return "wait_for_step"
            else:
                # Step is done for rover
                print('Skipping drone step (action is 0)')
                # await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_finished_step')
                return "follow_route"
        cust = self.customers[action - 1]
        target_x = cust['x'] * 10
        target_y = cust['y'] * 10
        print(f"Next target: {target_x}, {target_y}")
        moving = asyncio.ensure_future(rover.goto_coordinates(self.start_pos + VectorNED(target_y, -target_x, 0)))
        self.action_idx += 1

        await moving
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_finished_step')

        return "wait_for_step"

    @state(name="park")
    async def park(self, rover: Drone):
        print('Parking...')
        await asyncio.ensure_future(rover.goto_coordinates(self.takeoff_pos))
        await rover.land()
        print('Parked!')

