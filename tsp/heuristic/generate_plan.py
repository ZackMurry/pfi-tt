import torch
from torch import nn
import numpy as np
from NetworkDisruptionEnv import NetworkDisruptionEnv
# from LiveNetDisEnv import LiveNetDisEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import tianshou as ts
from tianshou.policy import BasePolicy
from tianshou.data import Batch

gym.envs.register(
    #  id='SimpleHeuristicTSPEnv-v0',
    #  entry_point=SimpleHeuristicTSPEnv,
     id='NetworkDisruptionEnv-v0',
    #  entry_point=LiveNetDisEnv,
    entry_point=NetworkDisruptionEnv,
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

env = FlattenObservation(NetworkDisruptionEnv())
# env.set_draw_all(True)
env.set_disruptions_enabled(False)
state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")

model = Net(state_shape, action_shape)
model.load_state_dict(torch.load("netdis_policy_5.pth"))
# model.load_state_dict(torch.load("policy.pth"))

print('Loaded model!')

class Net(nn.Module):
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

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(
    model=model,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)
policy.eval()
#policy = EvalPolicy(model, env.action_space)

# state, info = env.reset()
# print(state)
# state_tensor = Batch(obs=[state])
# setattr(state_tensor, 'info', {})

# todo: repeat n times and pick the fastest

with torch.no_grad():
    min_time = 999
    max_custs = 0
    for i in range(100):
        state, info = env.reset()
        env.set_disruptions_enabled(False)
        print(state)
        state_tensor = Batch(obs=[state])
        setattr(state_tensor, 'info', {})
        is_done = False
        num_steps = 0
        while not is_done:
            # print(state_tensor)
            result = policy(state_tensor)
            action = result.act[0]
            # print(env.planned_route)
            next_state, reward, done, truncated, info = env.step(action)
            is_done = done
            # print('Done?', is_done)
            state_tensor = Batch(obs=[next_state])
            setattr(state_tensor, 'info', {})
            # print('Action: ', action - 1, 'Reward: ', reward)
        print('Total time: ', env.total_time, '; num steps: ', num_steps)
        max_custs = max(max_custs, len(env.customers))
        if len(env.planned_route) == max_custs and env.total_time < min_time:
            print('Drawing new best!')
            env.draw_env()
            min_time = env.total_time
        elif False:
            env.draw_env()

        print('Min time: ', min_time, ' for ', max_custs)


# collector = ts.data.Collector(policy, env)


# collector.reset()

# result = collector.collect(n_episode=10)

print(f"result: {result}")
# print(f"Average reward: {result['rew']:.2f}, Total steps: {result['n/st']}")

