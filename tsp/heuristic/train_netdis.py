# import or_gym
# from or_gym.utils import create_env
#from SimpleHeuristicTSPEnv import SimpleHeuristicTSPEnv, save_logs
# from HeuristicTSPEnv import HeuristicTSPEnv, save_logs
from GoldwaterEnv import GoldwaterEnv
from NetworkDisruptionEnv import NetworkDisruptionEnv, save_logs
# from LiveNetDisEnv import LiveNetDisEnv, save_logs
from DisruptedScenario import DisruptedScenario
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import matplotlib.pyplot as plt
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import torch


num_train_envs = 4


# torch.set_default_tensor_type(torch.cuda.FloatTensor)

gym.envs.register(
    #  id='SimpleHeuristicTSPEnv-v0',
    #  entry_point=SimpleHeuristicTSPEnv,
     id='NetworkDisruptionEnv-v0',
    #  entry_point=LiveNetDisEnv,
    # entry_point=NetworkDisruptionEnv,
    entry_point=GoldwaterEnv,
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

train_envs = ts.env.SubprocVectorEnv([lambda: FlattenObservation(gym.make('NetworkDisruptionEnv-v0')) for _ in range(num_train_envs)])
test_envs = ts.env.SubprocVectorEnv([lambda: FlattenObservation(gym.make('NetworkDisruptionEnv-v0')) for _ in range(1)])

# train_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(num_train_envs)])
# test_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTruckDroneEnv-v0')) for _ in range(1)])
# train_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTSPEnv-v0')) for _ in range(num_train_envs)])
# test_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(gym.make('HeuristicTSPEnv-v0')) for _ in range(1)])

print('Saving scenario to file')
DisruptedScenario().export()

import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.device = device
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
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

env = FlattenObservation(GoldwaterEnv())
# env = FlattenObservation(HeuristicTSPEnv())

state_shape = env.observation_space.shape or env.observation_space.n
print(f'State shape: {state_shape}')
action_shape = env.action_space.shape or env.action_space.n
print(f"action shape: {action_shape}")
print(f"np.prod: {np.prod(action_shape)}")

# 1. Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Move the network to GPU
net = Net(state_shape, action_shape, device)
net = net.to(device)

# 3. Verify it's on GPU
print(f"Model on GPU: {next(net.parameters()).is_cuda}")

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
    target_update_freq=320,
).to(device)

# print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**2,1), 'MB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**2,1), 'MB')

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, num_train_envs), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

import os
log_path = os.path.join("dqn")
writer = SummaryWriter(log_path)
writer.add_text("args", "[]")
logger = TensorboardLogger(writer)

def train_callback(epoch, env_step):
    if env_step < 10_000:
        eps = 1.0
    else:
        # linear decay
        t = min(1.0, (env_step - 10_000) / 200_000)
        eps = 1.0 + t * (0.1 - 1.0)
    policy.set_eps(max(0.05, eps))
    # print(f"\n{'='*50}")
    # print(f"Epoch {epoch} - Step {env_step}")
    # print(f"{'='*50}")

def test_callback(epoch, env_step):
    policy.set_eps(0.05)
    
    # Collect test episodes
    test_result = test_collector.collect(n_episode=10, render=False)
    print(test_result)
    
    # Extract metrics from info dict
    if hasattr(test_result, 'info'):
        total_served = 0
        total_late = 0
        total_violations = 0
        n_episodes = len(test_result.info.get('customers_served', []))
        
        for i in range(n_episodes):
            total_served += test_result.info.get('customers_served', [0])[i]
            total_late += test_result.info.get('late_deliveries', [0])[i]
            total_violations += test_result.info.get('disrupted_violations', [0])[i]
        
        if n_episodes > 0:
            print(f"\nEpoch {epoch} Test Results:")
            print(f"  Avg Customers Served: {total_served/n_episodes:.2f}/12")
            print(f"  Service Rate: {total_served/(n_episodes*12)*100:.1f}%")
            print(f"  Avg Late Deliveries: {total_late/n_episodes:.2f}")
            print(f"  Avg Disrupted Violations: {total_violations/n_episodes:.2f}")

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    # max_epoch=60, 
    # step_per_epoch=10000,
    step_per_collect=16,
    max_epoch=300,
    step_per_epoch=5000,
    episode_per_test=20,  # Faster testing
    # episode_per_test=100, 
    update_per_step=0.1, batch_size=64,
    train_fn=train_callback,
    # test_fn=test_callback,  # Use custom test function
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    resume_from_log=True,
    stop_fn=lambda mean_rewards: mean_rewards >= 1000,
    logger=logger
).run()
print(f'Finished training!')

# save_logs()
env.reset()

policy.eval()
policy.set_eps(0.00)
# collector = ts.data.Collector(policy, env, exploration_noise=True)
# collector.collect(n_episode=5, render=1)

torch.save(policy.model.state_dict(), 'netdis_policy.pth')

# env.use_dataset = True
# collector.collect(n_episode=1, render=0)

