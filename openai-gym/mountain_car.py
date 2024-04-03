import gym
import matplotlib.pyplot as plt
from time import sleep
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np

def run_env():
  num_envs = 1
  print("Number of Envs: {}".format(num_envs))
  envs = [lambda: gym.make('MountainCar-v0') for i in range(num_envs)]
  envs = SubprocVecEnv(envs)
  
  obs = envs.reset()
  
  # ooos = one_obs.observation_space
  # print("Shape of one env: {}".format(one_obs.shape))
  # Velocity and position of the car
  # print("Observation space: {}".format(ooos))
  # print("Bounds: {} -> {}".format(ooos.low, ooos.high))

  print("Observation space: {}".format(envs.observation_space))
  print("Action space: {}".format(envs.action_space))

  # Take the action and get the new observation space
  # new_obs, reward, terminated, truncated, info = env.step(random_action)
  # print("New observation: {}".format(new_obs))

  num_steps = 1500
  print(dir(envs.observation_space))
  q_table = np.zeros([envs.observation_space.n, env.action_space.n])
  alpha = 0.1
  gamma = 0.6
  epsilon = 0.1

  for i in range(1,100001):
    epochs = 0
    state = envs.reset()
    done = False
    while not done:
      action = None
      if random.uniform(0,1) < epsilon:
        action = [envs.action_space.sample() for i in range(num_envs)] # 0=left,1=none,2=right
      else:
        action = np.argmax(q_table[state])
      
      next_state, reward, done, info = envs.step(action)
      old_value = q_table[state, action]
      next_max = np.max(q_table[next_state])
      new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
      q_table[state, action] = new_value
      state = next_state
      epochs += 1
    if i % 100 == 0:
      clear_output(wait=True)
      print(f"Episode: {i}")

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


