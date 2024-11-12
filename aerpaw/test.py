from aerpawlib.runner import StateMachine, entrypoint, state, timed_state
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio

class MyScript(StateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.path = []
        self.path_idx = 0
        file_name = '/root/export_2000.path'
        print(f"Reading data from {file_name}...")
        with open(file_name, 'r') as file:
            i = -1
            for line in file:
                i += 1
                if i % 20 == 0: # Prune out excessive data
                    # Split each line by whitespace and convert the values to integers or floats as needed
                    x, y = map(float, line.split())
                    self.path.append(((x - 100) / 4, (y - 100) / 4))
        print('Finished reading data!')
        
    @state(name="start", first=True)
    async def start(self, drone: Drone):
        print('Taking off...')
        await drone.takeoff(50)
        print('Finished taking off! Reached 50m')
        self.start_pos = drone.position
        print(f"Start pos: {self.start_pos}")
        print('Following route...')
        return "follow_route"

    @state(name="follow_route")
    async def follow_route(self, drone: Drone):
        # go north and continually log the vehicle's position
        target_x, target_y = self.path[self.path_idx]
        print(f"Next target: {target_x}, {target_y}")
        moving = asyncio.ensure_future(drone.goto_coordinates(self.start_pos + VectorNED(-target_y, -target_x, 0)))
        self.path_idx += 1

        while not moving.done():
            #print(f"pos: {drone.position}")
            await asyncio.sleep(0.1)

        await moving
        if self.path_idx >= len(self.path):
            return "land"
        else:
            return "follow_route"

    @state(name="land")
    async def land(self, drone: Drone):
        print('Landing...')
        await drone.land()
        print('Landed...')

