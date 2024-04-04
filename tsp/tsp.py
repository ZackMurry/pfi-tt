# import or_gym
# from or_gym.utils import create_env
from TSPEnv import TSPEnv, save_steps_log, save_dist_log, save_solved_log
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# import gi
# import ray
# from ray.rllib.algorithms import ppo
import tianshou as ts

# print([7+1]+[100]*7*2)

gym.envs.register(
     id='TSPEnv-v0',
     entry_point=TSPEnv,
     max_episode_steps=50,
     kwargs={},
)

print(ts.__version__)
train_envs = ts.env.DummyVectorEnv([lambda: gym.make('TSPEnv-v0') for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('TSPEnv-v0') for _ in range(100)])

import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

env = TSPEnv()


# env.render()

# plt.pause(10)

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=2e-3)

policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)

train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=25, 
    step_per_epoch=10000,
    step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold
).run()
print(f'Finished training! Took {result["duration"]}')

save_steps_log()
save_dist_log()
save_solved_log()

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=5, render=1)


# ray.init()
# algo = ppo.PPO(env=TSPEnv, config={})

# for i in range(1000):
#     print(f"Training iteration {i}")
#     result = algo.train()
#     print(f"Reward: {result['episode_reward_mean']}")


# print(np.version.version)

# env_config = {}

# # env_name = 'TSP-v0'
# # #create_env('tsp-v1')
# # env = or_gym.make(env_name, env_config=env_config)
# # print(dir(env))


# # Environment and RL Configuration Settings
# env = TSPEnv()
# print(env.action_space)

# state = env.reset()
# for i in range(1000):
#     action = env.action_space.sample()
#     print(action)
#     state, reward, done, _ = env.step(action)
#     print(reward)
#     env.plot_network()


