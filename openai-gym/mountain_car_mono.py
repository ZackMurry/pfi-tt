import gym
import matplotlib.pyplot as plt
from time import sleep
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np
import random

def state_to_index(q_table, state, env):
  low0 = env.observation_space.low[0]
  low1 = env.observation_space.low[1]
  return q_table[round(100 * (state[0] - low0))][round(100 * (state[1] - low1))]

def run_env():
  num_envs = 1
  print("Number of Envs: {}".format(num_envs))
  env = gym.make('MountainCar-v0')
  
  obs = env.reset()
  
  # ooos = one_obs.observation_space
  # print("Shape of one env: {}".format(one_obs.shape))
  # Velocity and position of the car
  # print("Observation space: {}".format(ooos))
  # print("Bounds: {} -> {}".format(ooos.low, ooos.high))

  print("Observation space: {}".format(env.observation_space))
  print("Action space: {}".format(env.action_space))

  # Take the action and get the new observation space
  # new_obs, reward, terminated, truncated, info = env.step(random_action)
  # print("New observation: {}".format(new_obs))

  num_steps = 1500
  print("Low: {}".format(env.observation_space.low))
  print("High: {}".format(env.observation_space.high))
  low0 = env.observation_space.low[0]
  low1 = env.observation_space.low[1]
  dist0 = env.observation_space.high[0] - low0
  dist1 = env.observation_space.high[1] - low1
  vals0 = round(dist0 * 100)
  vals1 = round(dist1 * 100)
  q_table = np.zeros([vals0, vals1, env.action_space.n])
  alpha = 0.1
  gamma = 0.6
  epsilon = 0.1
  env.render_mode = 'human'

  for i in range(1,100001):
    epochs = 0
    state = env.reset()
    done = False
    max_reward = 0
    max_height = 0
    while not done:
      action = None
      if random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(state_to_index(q_table, state, env))
      
      next_state, reward, done, info = env.step(action)
      print(next_state)
      max_height = max(max_height, next_state[1])
      reward = max_height
      print(reward)
      old_value = state_to_index(q_table, state, env)[action]
      next_max = np.max(state_to_index(q_table, next_state, env))
      new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
      state_to_index(q_table, state, env)[action] = new_value
      state = next_state
      epochs += 1
      if reward > max_reward:
        max_reward = reward
      if i % 1000 == 0:
        env.render()
    if i % 1000 == 0:
      print(f"Episode: {i}")
      print(f"Reward: {max_reward}")

  # for step in range(num_steps):
  #   action = [envs.action_space.sample() for i in range(num_envs)] # 0=left,1=none,2=right
  #   obs, reward, done, info = envs.step(action)
  #   envs.render()
  #   sleep(0.001)
  #   if True in done:
  #     envs.reset()
  # envs.close()
  
  
if __name__ == '__main__':
  run_env()


