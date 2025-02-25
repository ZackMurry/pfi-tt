"""
zmq_preplanend_orbit mission

there are two drones and a ground station

one drone, the tracer, follows a preplanned trajectory file
the other drone, the orbiter, will orbit the tracer at each waypoint

the ground coordinator will read from the file and issue controls to each drone (ex: go to this waypoint or orbit this waypoint)

the pattern proposed in this script is to have each "command" be a state on individual drones
thus, to make a drone do something, we transition their state from the central controller
"""

import asyncio
import datetime
import re
import csv

from aerpawlib.external import ExternalProcess
from aerpawlib.runner import ZmqStateMachine, state, background, in_background, timed_state, at_init, sleep, expose_field_zmq
from aerpawlib.util import Coordinate, Waypoint, read_from_plan_complete
from aerpawlib.vehicle import Drone
import torch
from torch import nn
import numpy as np
from LiveNetDisEnv import LiveNetDisEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch

gym.envs.register(
    #  id='SimpleHeuristicTSPEnv-v0',
    #  entry_point=SimpleHeuristicTSPEnv,
     id='NetworkDisruptionEnv-v0',
     entry_point=LiveNetDisEnv,
    # id='HeuristicTSPEnv-v0',
    # entry_point=HeuristicTSPEnv,
    max_episode_steps=50,
    kwargs={}
)

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


ZMQ_ROVER = "ROVER"
ZMQ_DRONE = "DRONE"

# todo: we need to minimize deviation from original plan

class GroundCoordinatorRunner(ZmqStateMachine):
  
  def __init__(self):
    print('Creating runner....')
    self.rover_taken_off = False
    self.drone_taken_off = False
    self.rover_finished_step = False
    self.drone_finished_step = False
    self.rover_waiting = False
    self.drone_waiting = False
    self.drone_landed = False
    self.rover_parked = False
    self.waiting_for_ping = False
    self.act_will_flip_dwt = False

    print("Reading plan...")
    self.actions = []
    self.rover_idx = -1
    self.drone_idx = -1
    self.drone_dests = []
    self.drone_dest_idx = 0

    plan_file_name = '/root/netdis.plan'
    print(f"Reading plan from {plan_file_name}...")
    with open(plan_file_name, 'r') as file:
        n = int(file.readline()) # num actions
        self.actions = list(map(int, file.readline().split()))
        print(f"actions: {self.actions}")
        m = int(file.readline()) # num drone dests
        self.drone_dests = list(map(int, file.readline().split()))
        print(f"drone_dests: {self.drone_dests}")
    
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

    self.env = FlattenObservation(LiveNetDisEnv())
    self.env.draw_all = True
    state_shape = self.env.observation_space.shape or self.env.observation_space.n
    print(f'State shape: {state_shape}')
    action_shape = self.env.action_space.shape or self.env.action_space.n
    print(f"action shape: {action_shape}")

    self.model = Net(state_shape, action_shape)
    # model.load_state_dict(torch.load("good_netdis_policy.pth"))
    self.model.load_state_dict(torch.load("netdis_policy_5.pth"))
    print('Loaded model!')

  @state(name="take_off", first=True)
  async def take_off(self, _):
    print("Sleeping 5s before takeoff...")
    await asyncio.sleep(5)
    print("Taking off...")
    await self.transition_runner(ZMQ_ROVER, "take_off")
    await self.transition_runner(ZMQ_DRONE, "take_off")
    print("Take off sent")
    return "await_taken_off"
  
  @state(name="await_taken_off")
  async def await_taken_off(self, _):
    # wait for both drones to finish taking off
    # this will be done by waiting for two flags to be set; each flag is set by transitioning to a special state
    if not (self.rover_taken_off and self.drone_taken_off):
      return "await_taken_off"
    print('Both taken off!')
    self.rover_finished_step = True
    self.drone_finished_step = True
    return "wait_for_step"

  @state(name="callback_rover_taken_off")
  async def callback_rover_taken_off(self, _):
    self.rover_taken_off = True
    print('Rover taken off!')
    return "await_taken_off"

  @state(name="callback_drone_taken_off")
  async def callback_drone_taken_off(self, _):
    self.drone_taken_off = True
    print('Drone taken off!')
    return "await_taken_off"
  
  def drone_with_truck(self, idx):
    idx = min(idx, len(self.actions))
    i = 0
    dwt = True
    while i < idx:
      if self.actions[i] == 0:
        dwt = not dwt
      i += 1
    return dwt

  def get_drone_dest(self, idx):
    i = 0
    dwt = True
    dest_idx = 0
    while i < idx:
      if self.actions[i] == 0:
        if not dwt:
          dest_idx += 1 # Increment dest index upon landing from previous trip
        dwt = not dwt
      i += 1
    return dest_idx
  
  @state(name="next_waypoint")
  async def next_waypoint(self, _):
    print('Sending STEP to rover and drone')


    # When dwt: truck and drone must move in sync
    # When not dwt:
    # - truck may move freely until next 0
    # - drone moves to drone_dest and then next 0

    await self.next_waypoint_rover(None)
    await self.next_waypoint_drone(None)
    
    return "wait_for_step"

  @state(name="next_waypoint_rover")
  async def next_waypoint_rover(self, _):
    self.rover_finished_step = False
    print('Next waypoint rover!')
    self.rover_idx += 1
    if self.rover_idx >= len(self.actions):
      # Park
      await self.transition_runner(ZMQ_DRONE, 'step')
      return

    dwt = self.drone_with_truck(self.rover_idx)
    act = self.actions[self.rover_idx]

    if act == 0 and dwt: # If drone leaving truck, just skip the action
      self.rover_idx += 1

    # But don't skip drone coming back!

      # act = self.actions[self.rover_idx]
      # dwt = False
    # elif not dwt and act != 0: # If truck+drone separate and not landing
    #   # Skip to landing
    #   while self.rover_idx < len(self.actions) and act != 0:
    #     self.rover_idx += 1
    #   if self.rover_idx == len(self.actions):
    #     # Park
    #     await self.transition_runner(ZMQ_DRONE, 'step')
    #     return

    await self.transition_runner(ZMQ_ROVER, 'step')
    return "wait_for_step"
  
  @state(name="next_waypoint_drone")
  async def next_waypoint_drone(self, _):
    self.drone_finished_step = False
    print('Next waypoint drone!')

    self.drone_idx += 1 # Advance to next state
    
    if self.drone_idx >= len(self.actions):
      # Park
      await self.transition_runner(ZMQ_DRONE, 'step')
      return "wait_for_step"


    dwt = self.drone_with_truck(self.rover_idx) # Is drone already with truck?

    if not dwt: # If drone is not with truck, skip over rover actions
      while self.drone_idx < len(self.actions) and self.actions[self.drone_idx] != 0:
        self.drone_idx += 1
    
    await self.transition_runner(ZMQ_DRONE, 'step')
    return "wait_for_step"


  def get_return_location_for_drone_at_index(self, d_idx):
    if d_idx >= len(self.actions):
      return None
    
    truck_location = None
    with_truck = True
    for i in range(len(self.actions)):
      if self.actions[i] == 0:
        with_truck = not with_truck
      else:
        truck_location = self.actions[i]
      if i > d_idx and with_truck:
        return truck_location

    assert False
    return truck_location

  def get_drone_next_by_index(self, _, idx):
    # This is a pure function -- no changes to the state
    # -1: land
    if idx >= len(self.actions):
      return -1 # Land
    
    dwt = self.drone_with_truck(idx)
    action = self.actions[idx]
    if action == 0: # Drone operation
      if dwt: # Drone is leaving truck
        dest_idx = self.get_drone_dest(self.drone_idx)
        return self.drone_dests[dest_idx]
      else: # Drone is returning to truck
        assert False, "drone had to wait"
        return self.get_return_location_for_drone_with_idx(idx) #wait!
    elif dwt:
      return action
    else:
      return self.get_drone_next_by_index(None, idx + 1)

  @expose_field_zmq(name="drone_next")
  async def get_drone_next(self, _):
    return self.get_drone_next_by_index(None, self.drone_idx)
  
  def get_rover_next_by_index(self, _, idx):
    # This is a pure function -- no changes to the state
    # -1: land
    # 0: wait
    if idx >= len(self.actions):
      return -1 # Park

    dwt = self.drone_with_truck(idx)
    action = self.actions[idx]
    if action == 0: # Drone operation
      if not dwt:
        # Wait for drone to meet truck
        assert False, "rover had to wait (1)"
        return 0
      else:
        # Sending out drone
        assert False, "rover had to wait (2)"
        return self.get_rover_next_by_index(None, idx + 1)
      
    # If not a drone action, go to the indicated customer
    return action

  @expose_field_zmq(name="rover_next")
  async def get_rover_next(self, _):
    return self.get_rover_next_by_index(None, self.rover_idx)


  @state(name="wait_for_step")
  async def wait_for_step(self, _):
    print('Waiting for step...')
    if self.rover_parked and self.drone_landed:
      print('Drone landed and rover parked!')
      return
    
    if self.rover_idx == self.drone_idx and self.rover_finished_step and self.drone_finished_step:
      # Simple case: drone and rover finished moving together
      print('drone and rover finished moving together')
      return "next_waypoint"
    

    dwt = self.drone_with_truck(self.rover_idx) # Drone is with truck before this move?

    # Case: drone is out, rover finished step and is waiting for drone
    if not dwt and self.actions[self.rover_idx] == 0 and self.rover_finished_step:
      # Do nothing? Waiting for next update
      print('Drone is out, rover finished step and is waiting for drone')

    # Case: drone is out, rover finished step
    if not dwt and self.actions[self.rover_idx] != 0 and self.rover_finished_step:
      print('Drone is out, rover finished step')
      return 'next_waypoint_rover'

    # Case: drone is out and finished delivery
    d_dwt = self.drone_with_truck(self.drone_idx)
    if not d_dwt and self.drone_finished_step: # todo
      print('drone is out and finished delivery')

    if self.drone_waiting and self.rover_finished_step:
      if self.rover_idx == self.drone_idx:
        print('drone waiting, rover finished, same idx')
      else:
        print('drone waiting, rover finished, different idx')

    

    return "wait_for_step"
    
      
  
  @state(name="callback_rover_finished_step")
  async def callback_rover_finished_step(self, _):
    print("callback_rover_finished_step")
    self.rover_finished_step = True
    return "wait_for_step"

  @state(name="callback_drone_finished_step")
  async def callback_drone_finished_step(self, _):
    print("callback_drone_finished_step")
    self.drone_finished_step = True
    return "wait_for_step"

  @state(name="callback_drone_landed")
  async def callback_drone_landed(self, _):
    print("Drone has landed!")
    self.drone_landed = True
    return "wait_for_step"

  @state(name="callback_rover_parked")
  async def callback_rover_parked(self, _):
    print("Rover has parked!")
    self.rover_parked = True
    return "wait_for_step"


