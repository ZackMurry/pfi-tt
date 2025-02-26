from aerpawlib.runner import ZmqStateMachine, entrypoint, state, timed_state
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit

print('Starting...')

ZMQ_COORDINATOR = 'COORDINATOR'

class DQNRover(ZmqStateMachine):

    def __init__(self):
        print("Initializing rover...")
        self.start_pos = None
        self.takeoff_pos = None
        self._takeoff_ordered = False
        self._next_step = False
        self._dwt = True
        
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
        if not self._takeoff_ordered:
            print('Received premature step!')
            return "take_off"
        self._next_step = True
        return "wait_for_step"

    @state(name="follow_route")
    async def follow_route(self, rover: Drone):
        print('Following route!')
        self._next_step = False

        next_waypoint = await self.query_field(ZMQ_COORDINATOR, "rover_next")
        print('next_waypoint: ', next_waypoint)

        if next_waypoint == -1:
            return "park"
        
        if next_waypoint == 0:
            await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_finished_step')
            return "wait_for_step"

        cust = self.customers[next_waypoint - 1]
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
        await self.transition_runner(ZMQ_COORDINATOR, 'callback_rover_parked')
        await asyncio.ensure_future(rover.goto_coordinates(self.takeoff_pos))
        await rover.land()
        print('Parked!')

