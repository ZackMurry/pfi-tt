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

ZMQ_ROVER = "ROVER"
ZMQ_DRONE = "DRONE"

class GroundCoordinatorRunner(ZmqStateMachine):
  
  def __init__(self):
    print('Creating runner....')
    self._rover_taken_off = False
    self._drone_taken_off = False
    self._rover_finished_step = False
    self._drone_finished_step = False
    self._dwt = True
    self._rover_waiting = False
    self._drone_waiting = False
    self._drone_landed = False
    self._rover_parked = False
    self._waiting_for_ping = False

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
    if not (self._rover_taken_off and self._drone_taken_off):
      return "await_taken_off"
    return "next_waypoint"

  @state(name="callback_rover_taken_off")
  async def callback_rover_taken_off(self, _):
    self._rover_taken_off = True
    print('Rover taken off!')
    return "await_taken_off"

  @state(name="callback_drone_taken_off")
  async def callback_drone_taken_off(self, _):
    self._drone_taken_off = True
    print('Drone taken off!')
    return "await_taken_off"
  
  @state(name="next_waypoint")
  async def next_waypoint(self, _):
    print('Sending STEP to rover and drone')
    self._rover_finished_step = False
    self._drone_finished_step = False
    await self.transition_runner(ZMQ_ROVER, 'step')
    await self.transition_runner(ZMQ_DRONE, 'step')
    return "wait_for_step"

  @state(name="next_waypoint_rover")
  async def next_waypoint_rover(self, _):
    print('Sending STEP to rover while drone is waiting')
    self._rover_finished_step = False
    await self.transition_runner(ZMQ_ROVER, 'step')
    return "wait_for_step"

  @state(name="handle_network_outage")
  async def handle_network_outage(self, _):
    print('Sending STEP to rover while drone is waiting')
    self._rover_finished_step = False
    await self.transition_runner(ZMQ_ROVER, 'step')
    return "wait_for_step"
  
  @state(name="wait_for_step")
  async def wait_for_step(self, _):
    print('Waiting for step...')
    ping_req = asyncio.ensure_future(self.query_field(ZMQ_DRONE, 'ping'))
    if self._rover_parked and self._drone_landed:
      print('Drone landed and rover parked!')
      return
    if self._rover_finished_step and self._drone_finished_step:
      print('Both finished step!')
      self._rover_waiting = False
      self._drone_waiting = False
      return "next_waypoint"
    if self._drone_waiting and self._rover_waiting:
      print('Both waited for each other!')
      self._drone_waiting = False
      self._rover_waiting = False
      return "next_waypoint"
    # if self._rover_finished_step and not self._dwt and not self._rover_waiting and not self._drone_waiting:
    #   print('Rover finished step, drone is out, and rover is not waiting')
    #   return "next_waypoint_rover"
    if self._drone_waiting and self._rover_finished_step:
      return "next_waypoint_rover"
    if not self._dwt and self._rover_finished_step and not self._rover_waiting:
      return "next_waypoint_rover"
    await asyncio.sleep(0.5)
    print("is ping req done: " + str(ping_req.done()))
    if not ping_req.done():
      print('Network outage!')
      ping_req.cancel()
      return "handle_network_outage"
    return "wait_for_step"
  
  @state(name="callback_rover_finished_step")
  async def callback_rover_finished_step(self, _):
    print("callback_rover_finished_step")
    self._rover_finished_step = True
    return "wait_for_step"

  @state(name="callback_drone_finished_step")
  async def callback_drone_finished_step(self, _):
    print("callback_drone_finished_step")
    self._drone_finished_step = True
    self._dwt = True
    return "wait_for_step"

  @state(name="callback_drone_out")
  async def callback_drone_out(self, _):
    print("callback_drone_out")
    self._dwt = False
    self._drone_waiting = False
    return "wait_for_step"

  @state(name="callback_rover_wait_for_drone")
  async def callback_rover_wait_for_drone(self, _):
    print("Rover is waiting for drone to return!")
    self._rover_waiting = True
    return "wait_for_step"

  @state(name="callback_drone_wait_for_rover")
  async def callback_drone_wait_for_rover(self, _):
    print("Drone is waiting to meet rover")
    self._drone_waiting = True
    self._dwt = True
    return "wait_for_step"

  @state(name="callback_drone_landed")
  async def callback_drone_landed(self, _):
    print("Drone has landed!")
    self._drone_landed = True
    return "wait_for_step"

  @state(name="callback_rover_parked")
  async def callback_rover_parked(self, _):
    print("Rover has parked!")
    self._rover_parked = True
    return "wait_for_step"


