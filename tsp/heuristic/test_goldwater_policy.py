import torch
from torch import nn
import numpy as np
from GoldwaterEnv import GoldwaterEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
import tianshou as ts
from tianshou.data import Batch

gym.envs.register(
    id='NetworkDisruptionEnv-v0',
    entry_point=GoldwaterEnv,
    max_episode_steps=50,
    kwargs={}
)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        input_dim = np.prod(state_shape)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape))
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class NearestNeighborPolicy:
    """
    Baseline policy: Always serve the nearest unserved customer with the truck.
    Never uses drone, never rejects customers.
    """
    def __init__(self, env):
        self.env = env
        self.truck_x = 0
        self.truck_y = 0
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        return abs(x2 - x1) + abs(y2 - y1)
    
    def _parse_obs(self, obs):
        """Parse flattened observation back to dict structure"""
        # Observation structure (for 12 customers):
        # planned_route: 12 values
        # drone_route: 12 values
        # request: 4 values (x, y, deadline, disrupted)
        # customers.x: 12 values
        # customers.y: 12 values
        # customers.deadline: 12 values
        # customers.disrupted: 12 values
        # customers_added: 1 value
        
        idx = 0
        num_customers = 12
        
        planned_route = obs[idx:idx+num_customers]
        idx += num_customers
        
        drone_route = obs[idx:idx+num_customers]
        idx += num_customers
        
        request = {
            'x': int(obs[idx]),
            'y': int(obs[idx+1]),
            'deadline': int(obs[idx+2]),
            'disrupted': int(obs[idx+3])
        }
        idx += 4
        
        customers = {
            'x': obs[idx:idx+num_customers],
            'y': obs[idx+num_customers:idx+2*num_customers],
            'deadline': obs[idx+2*num_customers:idx+3*num_customers],
            'disrupted': obs[idx+3*num_customers:idx+4*num_customers]
        }
        idx += 4*num_customers
        
        customers_added = int(obs[idx])
        
        return planned_route, request, customers, customers_added
    
    def select_action(self, obs):
        """
        Nearest neighbor strategy:
        1. If customer is disrupted, reject it (action 0)
        2. Otherwise, append to end of truck route (action = current_route_length + 2)
        """
        planned_route, request, customers, customers_added = self._parse_obs(obs)
        
        # If customer is disrupted, reject it
        if request['disrupted'] == 1:
            return 0  # Reject action
        
        # Otherwise, add to end of truck route
        # Count non-zero entries in planned_route to get current length
        current_route_length = int(np.sum(planned_route > 0))
        
        # Action encoding: 0=reject, 1=drone, 2+=insert at position
        # To append to end: action = 2 + current_route_length
        action = 2 + current_route_length
        
        return action
    
    def reset(self):
        self.truck_x = 0
        self.truck_y = 0


def evaluate_policy(policy, env, num_episodes=100, policy_name="Policy"):
    """Evaluate a policy and return metrics"""
    metrics = {
        'customers_served': [],
        'late_deliveries': [],
        'disrupted_violations': [],
        'rewards': [],
        'service_rates': []
    }
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        if hasattr(policy, 'reset'):
            policy.reset()
        
        while not done:
            steps += 1
            
            if isinstance(policy, NearestNeighborPolicy):
                action = policy.select_action(obs)
            else:
                # For DQN policy
                state_tensor = Batch(obs=[obs])
                setattr(state_tensor, 'info', {})
                with torch.no_grad():
                    result = policy(state_tensor)
                    action = result.act[0]
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                metrics['customers_served'].append(info.get('customers_served', 0))
                metrics['late_deliveries'].append(info.get('late_deliveries', 0))
                metrics['disrupted_violations'].append(info.get('disrupted_violations', 0))
                metrics['rewards'].append(episode_reward)
                service_rate = info.get('customers_served', 0) / 12
                metrics['service_rates'].append(service_rate)
                break
    
    # Calculate statistics
    results = {
        'policy_name': policy_name,
        'num_episodes': num_episodes,
        'avg_customers_served': np.mean(metrics['customers_served']),
        'std_customers_served': np.std(metrics['customers_served']),
        'avg_service_rate': np.mean(metrics['service_rates']) * 100,
        'avg_late_deliveries': np.mean(metrics['late_deliveries']),
        'avg_disrupted_violations': np.mean(metrics['disrupted_violations']),
        'avg_reward': np.mean(metrics['rewards']),
        'std_reward': np.std(metrics['rewards'])
    }
    
    return results, metrics


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS: {results['policy_name']}")
    print("="*70)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Avg Customers Served: {results['avg_customers_served']:.2f} ± {results['std_customers_served']:.2f} / 12")
    print(f"Service Rate: {results['avg_service_rate']:.1f}%")
    print(f"Avg Late Deliveries: {results['avg_late_deliveries']:.2f}")
    print(f"Avg Disrupted Violations: {results['avg_disrupted_violations']:.2f}")
    print(f"Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print("="*70 + "\n")


def compare_policies(rl_results, baseline_results):
    """Compare RL policy against baseline"""
    print("\n" + "="*70)
    print("COMPARISON: RL vs Baseline")
    print("="*70)
    
    improvement_customers = (
        (rl_results['avg_customers_served'] - baseline_results['avg_customers_served']) 
        / baseline_results['avg_customers_served'] * 100
    )
    improvement_reward = (
        (rl_results['avg_reward'] - baseline_results['avg_reward']) 
        / abs(baseline_results['avg_reward']) * 100
    )
    
    print(f"Customers Served Improvement: {improvement_customers:+.1f}%")
    print(f"  RL: {rl_results['avg_customers_served']:.2f} / 12")
    print(f"  Baseline: {baseline_results['avg_customers_served']:.2f} / 12")
    
    print(f"\nReward Improvement: {improvement_reward:+.1f}%")
    print(f"  RL: {rl_results['avg_reward']:.2f}")
    print(f"  Baseline: {baseline_results['avg_reward']:.2f}")
    
    print(f"\nLate Deliveries:")
    print(f"  RL: {rl_results['avg_late_deliveries']:.2f}")
    print(f"  Baseline: {baseline_results['avg_late_deliveries']:.2f}")
    
    print(f"\nDisrupted Violations:")
    print(f"  RL: {rl_results['avg_disrupted_violations']:.2f}")
    print(f"  Baseline: {baseline_results['avg_disrupted_violations']:.2f}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Setup environment
    env = FlattenObservation(GoldwaterEnv())
    
    print('Starting evaluation...\n')
    
    # ==================== EVALUATE BASELINE ====================
    print("Evaluating Nearest-Neighbor Baseline...")
    baseline_policy = NearestNeighborPolicy(env)
    baseline_results, baseline_metrics = evaluate_policy(
        baseline_policy, env, num_episodes=100, policy_name="Nearest-Neighbor Baseline"
    )
    print_results(baseline_results)
    
    # ==================== EVALUATE RL POLICY ====================
    print("Evaluating RL Policy...")
    
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    
    model = Net(state_shape, action_shape)
    model.load_state_dict(torch.load("netdis_policy.pth"))
    print('Loaded RL model!\n')
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    rl_policy = ts.policy.DQNPolicy(
        model=model,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320
    )
    rl_policy.eval()
    rl_policy.set_eps(0.0)  # No exploration during evaluation
    
    rl_results, rl_metrics = evaluate_policy(
        rl_policy, env, num_episodes=100, policy_name="RL Policy (DQN)"
    )
    print_results(rl_results)
    
    # ==================== COMPARISON ====================
    compare_policies(rl_results, baseline_results)
    
    # ==================== SINGLE EPISODE VISUALIZATION ====================
    print("\nRunning single episode with RL policy for visualization...")
    env_viz = FlattenObservation(GoldwaterEnv())
    # env_viz.env.draw_all = True  # Enable drawing
    
    obs, info = env_viz.reset()
    done = False
    steps = 0
    total_reward = 0
    
    actions = []
    while not done:
        steps += 1
        state_tensor = Batch(obs=[obs])
        setattr(state_tensor, 'info', {})
        
        with torch.no_grad():
            result = rl_policy(state_tensor)
            action = result.act[0]
        actions.append(action - 1)
        
        obs, reward, done, truncated, info = env_viz.step(action)
        total_reward += reward
        print(f'Step {steps}: Action {action-1}, Reward {reward:.2f}')
    env_viz.env.render(save_path='viz.png')
    
    print(f'Actions: {actions}')
    print(f'Route: {env_viz.env.planned_route}')
    print(f'All customers: {env_viz.env.all_customers}')
    print(f'\nVisualization episode complete!')
    print(f'Total steps: {steps}')
    print(f'Total reward: {total_reward:.2f}')
    print(f'Customers served: {info["customers_served"]}/12')
