import torch
from torch import nn
import numpy as np
from LiveNetDisEnv import LiveNetDisEnv
from GoldwaterEnv import GoldwaterEnv
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
    entry_point=GoldwaterEnv,
    # id='HeuristicTSPEnv-v0',
    # entry_point=HeuristicTSPEnv,
    max_episode_steps=50,
    kwargs={}
)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        input_dim = np.prod(state_shape)
        
        self.model = nn.Sequential(
            # Wider first layer to capture input complexity
            nn.Linear(input_dim, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Light regularization
            
            # Maintain capacity
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Funnel down
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Linear(128, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

env = FlattenObservation(GoldwaterEnv())


print('Starting...')

ZMQ_COORDINATOR = 'COORDINATOR'

env.draw_all = True
state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")

model = Net(state_shape, action_shape)
# model.load_state_dict(torch.load("good_netdis_policy.pth"))
model.load_state_dict(torch.load("netdis_policy.pth"))

print('Loaded model!')

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

state, info = env.reset()
print(state)
state_tensor = Batch(obs=[state])
setattr(state_tensor, 'info', {})

with torch.no_grad():
    is_done = False
    steps = 0
    while not is_done:
        steps += 1
        result = policy(state_tensor)
        action = result.act[0]
        print(env.planned_route)
        next_state, reward, done, truncated, info = env.step(action)
        is_done = done
        print('Done?', is_done)
        state_tensor = Batch(obs=[next_state])
        setattr(state_tensor, 'info', {})
        print('Action: ', action - 1, 'Reward: ', reward)
    print('Num steps: ', steps)


# collector = ts.data.Collector(policy, env)


# collector.reset()

# result = collector.collect(n_episode=10)

# print(f"result: {result}")
# print(f"Average reward: {result['rew']:.2f}, Total steps: {result['n/st']}")

