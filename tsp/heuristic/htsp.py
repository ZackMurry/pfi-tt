# import or_gym
# from or_gym.utils import create_env
#from SimpleHeuristicTSPEnv import SimpleHeuristicTSPEnv, save_logs
# from HeuristicTSPEnv import HeuristicTSPEnv, save_logs
from HeuristicTruckDroneEnv import HeuristicTruckDroneEnv, save_logs
from TSPScenario import TSPScenario
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
import tianshou as ts
import torch


num_train_envs = 1


# torch.set_default_tensor_type(torch.cuda.FloatTensor)

gym.envs.register(
    #  id='SimpleHeuristicTSPEnv-v0',
    #  entry_point=SimpleHeuristicTSPEnv,
     id='HeuristicTruckDroneEnv-v0',
     entry_point=HeuristicTruckDroneEnv,
    # id='HeuristicTSPEnv-v0',
    # entry_point=HeuristicTSPEnv,
    max_episode_steps=50,
    kwargs={}
)

print(ts.__version__)
print(f"CUDA: {torch.cuda.is_available()}")
# import envpool

# train_envs = ts.env.ShmemVectorEnv([lambda: gym.make('HeuristicTSPEnv-v0') for _ in range(10)])
# test_envs = ts.env.ShmemVectorEnv([lambda: gym.make('HeuristicTSPEnv-v0') for _ in range(100)])
# train_envs = ts.env.SubprocVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTSPEnv-v0')) for _ in range(num_train_envs)])
# train_envs = ts.env.SubprocVectorEnv([lambda: gym.make('HeuristicTruckDroneEnv-v0') for _ in range(num_train_envs)])
# test_envs = ts.env.SubprocVectorEnv([lambda: gym.make('HeuristicTruckDroneEnv-v0') for _ in range(100)])

train_envs = ts.env.SubprocVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(num_train_envs)])
test_envs = ts.env.SubprocVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(1)])

# train_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(num_train_envs)])
# test_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(1)])
# train_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTSPEnv-v0')) for _ in range(num_train_envs)])
# test_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTSPEnv-v0')) for _ in range(1)])

print('Saving scenario to file')
TSPScenario().export()

import torch, numpy as np
from torch import nn

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

env = FlattenObservation(HeuristicTruckDroneEnv())
# env = FlattenObservation(HeuristicTSPEnv())

state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")
print(f"np.prod: {np.prod(action_shape)}")
net = Net(state_shape, action_shape)

# print(next(net.parameters()).is_cuda) # False
# net.to(device)
# print(next(net.parameters()).is_cuda) # True

optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# policy  = ts.policy.PPOPolicy(
print(f"action space: {env.action_space}")
print(f"sample: {env.action_space.sample()}")
policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)#.to(device)

# print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, num_train_envs), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=100, 
    step_per_epoch=10000,
    step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
).run()
print(f'Finished training! Took {result["duration"]}')

save_logs()

policy.eval()
policy.set_eps(0.00)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=5, render=1)

env.use_dataset = True
collector.collect(n_episode=1, render=0)

