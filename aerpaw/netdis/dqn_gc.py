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
import sys

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
            print("OBS", obs)
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
    self.requested_park = False
    self.drone_disrupted = False
    self.disrupted_custs = []

    print("Reading plan...")
    self.actions = []
    self.rover_idx = -1
    self.drone_idx = -1
    self.drone_dests = []
    self.drone_dest_idx = -1

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

    self.served_custs = []
    d_dest_idx = 0
    act_idx = 0
    s_dwt = True
    for act in self.actions:
      if act == 0:
        if s_dwt:
          d_cust = self.drone_dests[d_dest_idx]
          d_dest_idx += 1
          self.served_custs.append(d_cust)
        s_dwt = not s_dwt
      else:
        self.served_custs.append(act)

    print('Served custs: ', self.served_custs)

    self.env = FlattenObservation(LiveNetDisEnv())
    self.env.unwrapped.set_served_custs(self.served_custs)
    state_shape = self.env.observation_space.shape or self.env.observation_space.n
    print(f'State shape: {state_shape}')
    action_shape = self.env.action_space.shape or self.env.action_space.n
    print(f"action shape: {action_shape}")

    self.model = Net(state_shape, action_shape)
    # model.load_state_dict(torch.load("good_netdis_policy.pth"))
    self.model.load_state_dict(torch.load("netdis_policy_5.pth"))
    print('Loaded model!')
    optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    self.policy = ts.policy.DQNPolicy(
        model=self.model,
        optim=optim,
        action_space=self.env.action_space,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320
    )
    self.policy.eval()
    print('Created policy!')

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

    if act == 0:
      if dwt: # If drone leaving truck, just skip the action
        print('Rover is skipping drone step at idx ', self.rover_idx)
        self.rover_finished_step = True
        await self.next_waypoint_rover(None)
        return "wait_for_step"
      else: # Wait for drone
        print('Rover should wait for drone')
        self.rover_finished_step = True
        return "wait_for_step"



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

    print('sending STEP to rover')
    await self.transition_runner(ZMQ_ROVER, 'step')
    return "wait_for_step"
  
  @state(name="next_waypoint_drone")
  async def next_waypoint_drone(self, _):
    self.drone_finished_step = False
    print('Next waypoint drone!')

    self.drone_idx += 1 # Advance to next state
    
    if self.drone_idx >= len(self.actions):
      # Park
      # print('sending STEP to drone')
      # await self.transition_runner(ZMQ_DRONE, 'step')
      return "wait_for_step"


    dwt = self.drone_with_truck(self.drone_idx) # Is drone already with truck?
    print('next_waypoint_drone dwt?: ', dwt)

    if not dwt: # If drone is not with truck, skip over rover actions
      while self.drone_idx < len(self.actions) and self.actions[self.drone_idx] != 0:
        self.drone_idx += 1
    
    print('sending STEP to drone')
    self.drone_dest_idx += 1
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
      if i >= d_idx and with_truck:
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
        print('Drone is returning to truck at idx: ', idx)
        return self.get_return_location_for_drone_at_index(idx) #wait!
    elif dwt:
      return action
    else:
      return self.get_drone_next_by_index(None, idx + 1)

  @expose_field_zmq(name="drone_next")
  async def get_drone_next(self, _):
    dn = self.get_drone_next_by_index(None, self.drone_idx)
    print("sending drone_next: ", dn)
    return dn
  
  def get_rover_next_by_index(self, _, idx):
    # This is a pure function -- no changes to the state
    # -1: land
    # 0: wait
    if idx >= len(self.actions):
      print('Sending park!')
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
    rn = self.get_rover_next_by_index(None, self.rover_idx)
    print('sending rover_next: ', rn)
    return rn

  @state(name="recover_disruption")
  async def recover_disruption(self, _):
    print('Recovering from disruption!')
    rover_pos = self.actions[self.rover_idx]
    print('Rover position: ', rover_pos)
    print('Completed actions: ', self.actions[0:self.rover_idx+1])
    dest_idx = self.get_drone_dest(self.drone_idx)
    print('Disrupted dest idx: ', dest_idx)
    print('Disrupted customer: ', self.drone_dests[dest_idx])
    print('Drone customers served: ', self.drone_dests[0:dest_idx])
    revised_actions = self.actions[0:self.rover_idx+1]
    last_zero = len(revised_actions) - 1 - revised_actions[::-1].index(0)
    revised_actions.pop(last_zero)
    def last_non_zero_before(arr, index):
      for i in range(index - 1, -1, -1):  # Iterate backwards from index - 1
          if arr[i] != 0:
              return arr[i]
      return None  # Return None if no non-zero element is found
    last_nz = last_non_zero_before(revised_actions, last_zero)
    print('Last truck dest before drone [where is drone returning to?]: ', last_nz)
    revised_actions.append(last_nz)
    print('Revised actions: ', revised_actions)

    self.disrupted_custs.append(self.drone_dests[dest_idx])
    cur_served_custs = []
    d_dest_idx = 0
    act_idx = 0
    s_dwt = True
    for act in revised_actions:
      if act == 0:
        if s_dwt:
          d_cust = self.drone_dests[d_dest_idx]
          d_dest_idx += 1
          cur_served_custs.append(d_cust)
        s_dwt = not s_dwt
      elif act not in cur_served_custs:
        cur_served_custs.append(act)
    
    print('Current served custs: ', cur_served_custs)
    # probably don't need a global self.served_custs
    self.env.unwrapped.set_served_custs(self.served_custs)

    self.env.unwrapped.set_preset_route(revised_actions, self.drone_dests[0:dest_idx], cur_served_custs)

    # only consider nodes in self.actions -- we only have the packages for these nodes
    # (implicit) first meet up with truck to exchange packages for drone
    # --> therefore, we must end the "preset" path with truck==drone
    # then feed the remaining customers into the program in order
    # --> this ensures that our remaining path is still similar to the original one (due to influence of order)
    # TODO: generate next steps

    print('Served custs: ', self.served_custs)
    state, *rest = self.env.step(-10)
    print("Got sentinel step")
    print(state)
    state_tensor = Batch(obs=[state])
    setattr(state_tensor, 'info', {})

    with torch.no_grad():
        is_done = False
        steps = 0
        while not is_done:
            steps += 1
            result = self.policy(state_tensor)
            action = result.act[0]
            print('GC Planned route: ', self.env.unwrapped.planned_route)
            next_state, reward, done, truncated, info = self.env.step(action)
            is_done = done
            print('Done?', is_done)
            state_tensor = Batch(obs=[next_state])
            setattr(state_tensor, 'info', {})
            print('Action: ', action - 1, 'Reward: ', reward)
    print('Num steps: ', steps)
    print('Translated route', self.env.unwrapped.planned_route)
    new_route = self.env.unwrapped.get_planned_route()
    print('Untranslated new route', new_route)
    print('Translated drone route' ,self.env.unwrapped.drone_route)
    new_drone_route = self.env.unwrapped.get_drone_route()
    print('Untranslated drone dests', new_drone_route)
    print("Old actions ", self.actions)
    print("Cutoff at ", self.rover_idx)
    self.actions = new_route
    self.drone_dests = new_drone_route
    self.rover_finished_step = True
    self.drone_finished_step = True
    self.drone_disrupted = False
    self.rover_idx = len(revised_actions) - 2
    self.drone_idx = len(revised_actions) - 2

    print('new idx:', self.rover_idx)
    
    return "wait_for_step"



  @state(name="wait_for_step")
  async def wait_for_step(self, _):
    print('Waiting for step...')
    dwt = self.drone_with_truck(self.rover_idx) # Drone is with truck before this move?
    print('dwt: ', dwt)
    print('rover_idx: ', self.rover_idx)
    print('drone_idx: ', self.drone_idx)
    print('actions: ', self.actions)
    print('drone_dests: ', self.drone_dests)
    print('drone_finished: ', self.drone_finished_step)
    print('rover_finished: ', self.rover_finished_step)
    if self.rover_parked and self.drone_landed:
      print('Drone landed and rover parked!')
      return

    if self.drone_disrupted:
      print('Drone disrupted!')
      if self.rover_finished_step and self.drone_finished_step:
        self.drone_disrupted = False
        return "recover_disruption"
      else:
        print('Waiting for drone+rover to settle...')
        await asyncio.sleep(0.5)
        return "wait_for_step"

    if self.rover_idx >= len(self.actions) and self.drone_idx >= len(self.actions) and not self.requested_park:
      # Park
      print('Park!')
      await self.transition_runner(ZMQ_DRONE, 'step')
      await self.transition_runner(ZMQ_ROVER, 'step')
      self.requested_park = True
      return "wait_for_step"
    

    
    if self.rover_idx == self.drone_idx and self.rover_finished_step and self.drone_finished_step:
      # Simple case: drone and rover finished moving together
      print('drone and rover finished moving together')
      return "next_waypoint"
    


    # Case: drone is out, rover finished step and is waiting for drone
    if not dwt and (self.rover_idx >= len(self.actions) or self.actions[self.rover_idx] == 0) and self.rover_finished_step and self.drone_finished_step:
      print('Rover waiting for drone, so next_waypoint_drone')
      return 'next_waypoint_drone'

    # Case: drone is out, rover finished step
    if not dwt and (self.rover_idx >= len(self.actions) or self.actions[self.rover_idx] != 0) and self.rover_finished_step:
      print('Drone is out, rover finished step')
      if self.drone_finished_step:
        return 'next_waypoint'
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

    await asyncio.sleep(0.5)

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
    self.rover_finished_step = True
    return "wait_for_step"

  @state(name="callback_rover_parked")
  async def callback_rover_parked(self, _):
    print("Rover has parked!")
    self.rover_parked = True
    self.rover_finished_step = True
    return "wait_for_step"

  @state(name="callback_drone_disrupted")
  async def callback_drone_disrupted(self, _):
    print("Drone has been disrupted!")
    self.drone_disrupted = True
    return "wait_for_step"


