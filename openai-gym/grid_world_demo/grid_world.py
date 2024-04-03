from env.grid_world_env import GridWorldEnv
import numpy as np
import random

size = 5
env = GridWorldEnv(render_mode="human", size=5)
obs = env.reset()


# ax, ay, tx, ty, actions
q_table = np.zeros([size, size, size, size, env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1

avg_steps = 0
recent_steps = 0
for i in range(1,1000001):
  if i % 1000 == 0:
    avg_steps = recent_steps / 1000
    recent_steps = 0
    print(f"i: {i}, average steps: {avg_steps}")
    
  epochs = 0
  env.render_mode = "human" if i % 1000 == 0 else None
  obs, info = env.reset()
  done = False

  while not done and epochs < 300:
    # print("step: {}".format(epochs))
    action = None
    agent = obs['agent']
    target = obs['target_rel']
    qstate = q_table[agent[0]][agent[1]][target[0]][target[1]]
    if random.uniform(0,1) < epsilon:
      action = env.action_space.sample()
    else:
      action = np.argmax(qstate)
    
    next_state, reward, done, _, info = env.step(action)
    old_value = qstate[action]
    next_max = np.max(qstate)
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    qstate[action] = new_value
    obs = next_state
    epochs += 1
    if i % 1000 == 0:
      # print(f"Reward: {reward}, qstate: {qstate}")
      # print(target)
      env.render()
  
  recent_steps += epochs

env.close()
