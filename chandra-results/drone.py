from aerpawlib.runner import StateMachine, entrypoint, state, timed_state, background
from aerpawlib.vehicle import Drone
from aerpawlib.util import Coordinate, VectorNED
import asyncio
from sys import exit
import random
import torch
from torch import nn
import numpy as np
import time
import datetime
from interface import get_path


mock = False
chandra = True

GRID_SIZE = 15

class ChandraDrone(StateMachine):

    def __init__(self):
        print("Initializing drone...")
        self.start_pos = None
        self.total_dist = 0
        self.total_move_time = 0
        print('initialized')
        
    @background
    async def log_in_background(self, vehicle: Drone):
        print(f"vehicle pos: {vehicle.position}")
        await asyncio.sleep(1)
        
    @state(name="start", first=True)
    async def start(self, drone: Drone):
        print('Taking off...')
        if not mock:
            await drone.takeoff(55)
        print('Finished taking off! Reached 50m')
        self.start_pos = drone.position
        print(f"Start pos: {self.start_pos}")
        print("Sending callback to coordinator")
        print('Following route...')
        now = datetime.datetime.now()
        next_minute = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        seconds_to_sleep = (next_minute - now).total_seconds()
        print('sleeping for ', seconds_to_sleep)
        await asyncio.sleep(seconds_to_sleep)
        return "follow_route"

    @state(name="follow_route")
    async def follow_route(self, drone: Drone):
        print('following route!')
        self._next_step = False

        start = drone.position  # Must be lat/lon
        waypoint_x = 10
        waypoint_y = 10
        goal = self.start_pos + VectorNED(waypoint_y * 10, -waypoint_x * 10, 0)

        if chandra:
            # We force the RL path to operate on a fixed grid space
            grid_start = (0, 0)
            grid_goal = (GRID_SIZE - 1, GRID_SIZE - 1)

            def grid_to_latlon(gx, gy):
                frac_x = gx / (GRID_SIZE - 1)
                frac_y = gy / (GRID_SIZE - 1)
                print('frac_x: ', frac_x, ", frac_y: ", frac_y)
                print('lat range; ', goal.lat - start.lat, ", lon range: ", goal.lon - start.lon)

                lat = start.lat + frac_y * (goal.lat - start.lat)
                lon = start.lon + frac_x * (goal.lon - start.lon)
                return Coordinate(lat=lat, lon=lon, alt=start.alt)

            def generate_random_coordinates(N):
              # Generate coordinates in the range [0, 14], excluding (0, 0) and (14, 14)
              all_coords = [(x, y) for x in range(15) for y in range(15) if (x, y) not in [(0, 0), (14, 14)]]
              
              if N > len(all_coords):
                  raise ValueError(f"Cannot generate {N} unique coordinates from the available {len(all_coords)} points.")

              return random.sample(all_coords, N)
            # obstacles = generate_random_coordinates(2) # still empty unless you want to map no-fly zones etc.
            obstacles = [(12, 14), (3, 10), (12, 4), (3, 2), (0, 1), (12, 5)]
            print('obstacles', obstacles)

            path = get_path(grid_start, grid_goal, obstacles)
            print(path)

            num_waypoints = 10
            if len(path) <= num_waypoints:
                sampled_points = path
            else:
                indices = [round(i * (len(path) - 1) / (num_waypoints - 1)) for i in range(num_waypoints)]
                sampled_points = [path[i] for i in indices]
            sampled_points.append([14,14])
            
            print('samples', sampled_points)

            self.is_moving = True
            for (gx, gy) in sampled_points[1:]:  # skip first (it's current position)
                target = grid_to_latlon(gx, gy)
                print(f"Moving to ({gx},{gy}) => {target}")
                dist = drone.position.ground_distance(target)
                print(f"dist: {dist}")
                self.total_dist += dist
                t0 = time.time()
                await drone.goto_coordinates(target)
                self.total_move_time += time.time() - t0
                print('Reached target!')
                # await asyncio.sleep(3)
            
            print('Finished Chandra points')
        else:
            dist = drone.position.ground_distance(goal)
            print(f"dist: {dist}")
            self.total_dist += dist
            t0 = time.time()
            await drone.goto_coordinates(goal)
            self.total_move_time += time.time() - t0
            print('Reached target!')


        self.is_moving = False
        self.finished = True
        print('sending callback_drone_finished_step')

        return "land"

    @state(name="land")
    async def land(self, drone: Drone):
        print('Landing...')
        print(f"Move dist {self.total_dist}")
        print(f"Move time {self.total_move_time}")
        if not mock:
            await asyncio.ensure_future(drone.goto_coordinates(self.start_pos))
            await drone.land()
        print(f"Move dist {self.total_dist}")
        print(f"Move time {self.total_move_time}")
        print('Landed...')
    


