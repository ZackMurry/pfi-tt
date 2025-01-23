import torch
from torch import nn
import numpy as np
from NetworkDisruptionEnv import NetworkDisruptionEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import tianshou as ts
from tianshou.policy import BasePolicy

gym.envs.register(
    #  id='SimpleHeuristicTSPEnv-v0',
    #  entry_point=SimpleHeuristicTSPEnv,
     id='NetworkDisruptionEnv-v0',
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
state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")

model = Net(state_shape, action_shape)
model.load_state_dict(torch.load("netdis_policy.pth"))

print('Loaded model!')

class EvalPolicy(BasePolicy):
    def __init__(self, model, action_space):
        super().__init__(action_space=action_space)
        self.model = model

    def forward(self, batch, state=None, **kwargs):
        # Pass observations through the model to get actions
        obs = torch.tensor(batch.obs, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(obs)
            print(logits)
            print(logits[0])
            print(logits[0][0])
            actions = logits[0].argmax(dim=-1).numpy()  # Choose the action with the highest score
        return actions, state

    def learn(self):
      return

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(
    model=model,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)
#policy = EvalPolicy(model, env.action_space)

collector = ts.data.Collector(policy, env)

collector.reset()

result = collector.collect(n_episode=10)

print(f"result: {result}")
# print(f"Average reward: {result['rew']:.2f}, Total steps: {result['n/st']}")

