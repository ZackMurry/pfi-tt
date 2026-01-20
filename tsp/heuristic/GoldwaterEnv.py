import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
from types import SimpleNamespace
from math import pow, sqrt

class GoldwaterEnv(gym.Env):
    """
    RoutePEARL environment for training RL agents on network-aware
    truck-drone delivery route planning.
    """
    def __init__(self, num_customers=12, *args, **kwargs):
        # Configuration
        self.NUM_CUSTOMERS = num_customers
        self.DRONE_SPEED_FACTOR = 1.5  # Drones are faster than trucks
        self.MAX_T = 200  # Maximum time horizon
        self.MAX_X = 20
        self.MAX_Y = 20
        self.DISRUPTION_PROB = 0.2  # 20% of customers have network disruptions
        
        # Episode tracking
        self.episodes = 0
        self.step_count = 0
        self.spec = SimpleNamespace(reward_threshold=100)
        
        # Route state
        self.customers = []
        self.planned_route = []
        self.drone_route = []
        self.rejected = []
        self.drone_with_truck = True
        
        # Position tracking
        self.depot = {"x": 0, "y": 0}
        self.truck_x = 0
        self.truck_y = 0
        
        # Reward tracking
        self.customers_served = 0
        self.late_deliveries = 0
        self.disrupted_drone_deliveries = 0
        self.previous_route_time = 0  # For reward shaping
        
        # Define observation space
        max_queue = self.NUM_CUSTOMERS + 2
        self.observation_space = spaces.Dict({
            "planned_route": spaces.MultiDiscrete([max_queue] * self.NUM_CUSTOMERS),
            "drone_route": spaces.MultiDiscrete([max_queue] * self.NUM_CUSTOMERS),
            "request": spaces.Dict({
                "x": spaces.Discrete(self.MAX_X + 1),
                "y": spaces.Discrete(self.MAX_Y + 1),
                "deadline": spaces.Discrete(self.MAX_T + 1),
                "disrupted": spaces.Discrete(2)
            }),
            "customers": spaces.Dict({
                "x": spaces.MultiDiscrete([self.MAX_X + 1] * self.NUM_CUSTOMERS),
                "y": spaces.MultiDiscrete([self.MAX_Y + 1] * self.NUM_CUSTOMERS),
                "deadline": spaces.MultiDiscrete([self.MAX_T + 1] * self.NUM_CUSTOMERS),
                "disrupted": spaces.MultiDiscrete([2] * self.NUM_CUSTOMERS)
            }),
            "customers_added": spaces.Discrete(self.NUM_CUSTOMERS + 1)
        })
        
        # Define action space: reject (-1), drone (0), or insert at position [1, n]
        self.action_space = spaces.Discrete(2 + self.NUM_CUSTOMERS)
        
        self.reset()
    
    def _generate_customers(self):
        """Generate randomized customer locations and deadlines with controlled difficulty"""
        customers = []
        
        # Generate customers in clusters to ensure feasibility
        num_clusters = np.random.randint(2, 4)  # 2-3 clusters
        customers_per_cluster = self.NUM_CUSTOMERS // num_clusters
        
        for cluster_idx in range(num_clusters):
            # Random cluster center
            center_x = np.random.randint(3, self.MAX_X - 3)
            center_y = np.random.randint(3, self.MAX_Y - 3)
            
            for i in range(customers_per_cluster):
                # Generate customer near cluster center
                x = np.clip(center_x + np.random.randint(-4, 5), 1, self.MAX_X - 1)
                y = np.clip(center_y + np.random.randint(-4, 5), 1, self.MAX_Y - 1)
                
                # Deadline: guaranteed feasible + buffer
                min_time = self._manhattan_distance(0, 0, x, y)
                # Add buffer: 2x-3x the minimum time to ensure feasibility
                deadline = int(min_time * np.random.uniform(2.0, 3.5))
                
                # Controlled disruption probability
                disrupted = 1 if np.random.random() < self.DISRUPTION_PROB else 0
                
                customers.append({
                    "x": x,
                    "y": y,
                    "deadline": deadline,
                    "disrupted": disrupted
                })
        
        # Fill remaining customers if any
        while len(customers) < self.NUM_CUSTOMERS:
            x = np.random.randint(2, self.MAX_X - 2)
            y = np.random.randint(2, self.MAX_Y - 2)
            min_time = self._manhattan_distance(0, 0, x, y)
            deadline = int(min_time * np.random.uniform(2.0, 3.5))
            disrupted = 1 if np.random.random() < self.DISRUPTION_PROB else 0
            
            customers.append({
                "x": x,
                "y": y,
                "deadline": deadline,
                "disrupted": disrupted
            })
        
        # Shuffle to randomize order
        np.random.shuffle(customers)
        
        return customers
    
    def _calculate_scenario_difficulty(self):
        """Calculate difficulty score for current scenario (0=easy, 1=hard)"""
        if len(self.all_customers) == 0:
            return 0.5
        
        # Factor 1: Average distance from depot
        avg_distance = np.mean([
            self._manhattan_distance(0, 0, c['x'], c['y']) 
            for c in self.all_customers
        ])
        distance_score = min(avg_distance / 20.0, 1.0)  # Normalize to [0,1]
        
        # Factor 2: Deadline tightness (ratio of deadline to minimum time)
        tightness_scores = []
        for c in self.all_customers:
            min_time = self._manhattan_distance(0, 0, c['x'], c['y'])
            if min_time > 0:
                tightness = c['deadline'] / min_time
                tightness_scores.append(max(0, 1 - (tightness - 1) / 2))  # Lower is tighter
        avg_tightness = np.mean(tightness_scores) if tightness_scores else 0.5
        
        # Factor 3: Disruption rate
        disruption_rate = sum(c['disrupted'] for c in self.all_customers) / len(self.all_customers)
        
        # Combine factors (weighted average)
        difficulty = 0.3 * distance_score + 0.4 * avg_tightness + 0.3 * disruption_rate
        
        return difficulty
    
    def _get_action_mask(self):
        """
        Return boolean mask of valid actions.
        Action encoding: 0=reject, 1=drone, 2+=insert at position
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        # Reject is always valid
        mask[0] = 1
        
        # Drone is valid if we have a request
        if self.request_idx < len(self.all_customers):
            mask[1] = 1
        
        # Can insert at positions 0 through current_route_length
        # Action 2 = insert at position 0, Action 3 = insert at position 1, etc.
        max_insert_action = min(2 + len(self.planned_route) + 1, self.action_space.n)
        mask[2:max_insert_action] = 1
        
        return mask
        """
        Return boolean mask of valid actions.
        Action encoding: 0=reject, 1=drone, 2+=insert at position
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        # Reject is always valid
        mask[0] = 1
        
        # Drone is valid if we have a request
        if self.request_idx < len(self.all_customers):
            mask[1] = 1
        
        # Can insert at positions 0 through current_route_length
        # Action 2 = insert at position 0, Action 3 = insert at position 1, etc.
        max_insert_action = min(2 + len(self.planned_route) + 1, self.action_space.n)
        mask[2:max_insert_action] = 1
        
        return mask
    
    def _estimate_route_time(self):
        """Estimate total time for current route (used for reward shaping)"""
        if len(self.planned_route) == 0:
            return 0
        
        time = 0
        x, y = 0, 0
        
        drone_with_truck = True
        drone_idx = 0
        drone_time = 0
        drone_start_time = 0
        
        for dest in self.planned_route:
            if dest == 0:  # Drone action
                if drone_with_truck and drone_idx < len(self.drone_route):
                    cust = self.customers[self.drone_route[drone_idx] - 1]
                    drone_time = self._get_travel_time(x, y, cust, is_drone=True)
                    drone_start_time = time
                    drone_with_truck = False
                elif not drone_with_truck and drone_idx < len(self.drone_route):
                    cust = self.customers[self.drone_route[drone_idx] - 1]
                    return_time = self._get_travel_time(x, y, cust, is_drone=True)
                    drone_time += return_time
                    wait_time = max(0, drone_time - (time - drone_start_time))
                    time += wait_time
                    drone_idx += 1
                    drone_with_truck = True
                continue
            
            if dest > 0 and dest <= len(self.customers):
                cust = self.customers[dest - 1]
                travel_time = self._get_travel_time(x, y, cust)
                time += travel_time
                x, y = cust['x'], cust['y']
        
        return time
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points"""
        return abs(x2 - x1) + abs(y2 - y1)
    
    def _get_travel_time(self, x, y, customer, is_drone=False):
        """Get travel time to customer"""
        dist = self._manhattan_distance(x, y, customer['x'], customer['y'])
        if is_drone:
            return dist / self.DRONE_SPEED_FACTOR
        return dist
    
    def step(self, action):
        """Execute one step in the environment"""
        action -= 1  # Convert to: reject=-1, drone=0, positions=[1,n]
        self.step_count += 1
        
        reward = 0
        done = False
        
        # Get current request
        if self.request_idx >= len(self.all_customers):
            return self.state, 0, True, True, {"action_mask": self._get_action_mask()}
        
        request = self.all_customers[self.request_idx]
        
        # Process action
        if action == -1:  # Reject customer
            self.rejected.append(request)
            reward -= 3  # Increased penalty for rejecting
            
        elif action == 0:  # Send drone
            if request['disrupted']:
                # CRITICAL: Large penalty for sending drone to disrupted location
                reward -= 15  # Increased from 10
                self.disrupted_drone_deliveries += 1
                done = True
            else:
                # Reward for smart drone usage
                if self.drone_with_truck:
                    self.planned_route.append(0)  # Mark drone departure
                    self.drone_route.append(len(self.customers) + 1)
                    self.drone_with_truck = False
                    reward += 2  # Reward for using drone efficiently
                else:
                    self.planned_route.append(0)  # Mark drone return
                    self.drone_with_truck = True
                    reward += 2  # Reward for collecting drone
                
                self.customers.append(request)
        
        else:  # Add to truck route at position
            insert_pos = min(action, len(self.planned_route) + 1)
            self.planned_route.insert(insert_pos - 1, len(self.customers) + 1)
            self.customers.append(request)
            reward += 1.0  # Reduced from 1.5 to make truck less attractive
        
        # Estimate route efficiency (reward shaping)
        current_route_time = self._estimate_route_time()
        if self.previous_route_time > 0:
            # Reward if route is getting more efficient
            time_improvement = self.previous_route_time - current_route_time
            if time_improvement > 0:
                reward += min(time_improvement * 0.1, 1.0)  # Cap improvement bonus
            elif time_improvement < -5:  # Penalize if route gets much worse
                reward -= 0.5
        self.previous_route_time = current_route_time
        
        # Validate route and calculate delivery times
        route_valid, num_late = self._validate_route()
        
        if not route_valid:
            reward -= 8  # Increased penalty for invalid routes
            done = True
        
        # Stronger penalty for late deliveries
        if num_late > 0:
            reward -= num_late * 4  # Increased from 2
            self.late_deliveries += num_late
        
        # Move to next customer
        self.request_idx += 1
        
        # Check if all customers processed
        if self.request_idx >= len(self.all_customers):
            # Final reward based on customers served
            self.customers_served = len(self.customers)
            
            # Progressive reward for customers served
            reward += self.customers_served * 5  # Increased from 3
            
            # Big bonus for serving all customers on time
            if num_late == 0 and len(self.customers) == self.NUM_CUSTOMERS:
                reward += 20  # Increased from 10
            
            # Bonus for high service rate
            service_rate = self.customers_served / self.NUM_CUSTOMERS
            if service_rate >= 0.9:
                reward += 10
            elif service_rate >= 0.75:
                reward += 5
            
            done = True
        
        self._update_state()
        
        return self.state, reward, done, False, {
            "action_mask": self._get_action_mask(),
            "customers_served": self.customers_served,
            "late_deliveries": self.late_deliveries,
            "disrupted_violations": self.disrupted_drone_deliveries
        }
    
    def _validate_route(self):
        """Validate the current route and return (valid, num_late_deliveries)"""
        if len(self.planned_route) == 0:
            return True, 0
            
        time = 0
        x, y = 0, 0
        late_count = 0
        
        drone_with_truck = True
        drone_idx = 0
        drone_time = 0
        drone_start_time = 0
        drone_x, drone_y = 0, 0  # Track where drone was deployed from
        
        for i, dest in enumerate(self.planned_route):
            if dest == 0:  # Drone action
                if drone_with_truck:  # Send drone out
                    if drone_idx >= len(self.drone_route):
                        # Missing drone route entry - route is malformed
                        return False, late_count
                    
                    cust_idx = self.drone_route[drone_idx] - 1
                    if cust_idx < 0 or cust_idx >= len(self.customers):
                        # Invalid customer index
                        return False, late_count
                    
                    cust = self.customers[cust_idx]
                    drone_time = self._get_travel_time(x, y, cust, is_drone=True)
                    drone_start_time = time
                    drone_x, drone_y = x, y  # Remember where drone left from
                    
                    if time + drone_time > cust['deadline']:
                        late_count += 1
                    
                    drone_with_truck = False
                else:  # Collect drone
                    cust_idx = self.drone_route[drone_idx] - 1
                    if cust_idx < 0 or cust_idx >= len(self.customers):
                        return False, late_count
                        
                    cust = self.customers[cust_idx]
                    # Drone returns from customer back to current truck position
                    return_time = self._get_travel_time(x, y, cust, is_drone=True)
                    total_drone_time = drone_time + return_time
                    
                    # Truck waits if drone isn't back yet
                    elapsed_truck_time = time - drone_start_time
                    wait_time = max(0, total_drone_time - elapsed_truck_time)
                    time += wait_time
                    
                    drone_idx += 1
                    drone_with_truck = True
                continue
            
            # Truck delivery
            if dest <= 0 or dest > len(self.customers):
                # Invalid customer reference
                return False, late_count
                
            cust = self.customers[dest - 1]
            travel_time = self._get_travel_time(x, y, cust)
            time += travel_time
            
            if time > cust['deadline']:
                late_count += 1
            
            x, y = cust['x'], cust['y']
        
        # It's OK if drone isn't back yet - we'll just count it as incomplete
        # Don't invalidate the entire route for this
        
        return True, late_count
    
    def _update_state(self):
        """Update the observation state"""
        # Get next request
        if self.request_idx < len(self.all_customers):
            req = self.all_customers[self.request_idx]
        else:
            req = {"x": 0, "y": 0, "deadline": 0, "disrupted": 0}
        
        # Pad customer arrays
        custs = {"x": [], "y": [], "deadline": [], "disrupted": []}
        for cust in self.customers:
            custs['x'].append(cust['x'])
            custs['y'].append(cust['y'])
            custs['deadline'].append(cust['deadline'])
            custs['disrupted'].append(cust['disrupted'])
        
        while len(custs['x']) < self.NUM_CUSTOMERS:
            custs['x'].append(0)
            custs['y'].append(0)
            custs['deadline'].append(0)
            custs['disrupted'].append(0)
        
        # Pad routes
        planned = self.planned_route.copy()
        while len(planned) < self.NUM_CUSTOMERS:
            planned.append(0)
        
        drone_r = self.drone_route.copy()
        while len(drone_r) < self.NUM_CUSTOMERS:
            drone_r.append(0)
        
        self.state = {
            "request": req,
            "customers": {
                "x": np.array(custs['x']),
                "y": np.array(custs['y']),
                "deadline": np.array(custs['deadline']),
                "disrupted": np.array(custs['disrupted'])
            },
            "planned_route": np.array(planned),
            "drone_route": np.array(drone_r),
            "customers_added": len(self.customers)
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        if seed is not None:
            np.random.seed(seed)
        
        self.episodes += 1
        self.step_count = 0
        self.request_idx = 0
        
        # Generate new customers
        self.all_customers = self._generate_customers()
        
        # Reset state
        self.customers = []
        self.planned_route = []
        self.drone_route = []
        self.rejected = []
        self.drone_with_truck = True
        self.truck_x = 0
        self.truck_y = 0
        
        # Reset metrics
        self.customers_served = 0
        self.late_deliveries = 0
        self.disrupted_drone_deliveries = 0
        self.previous_route_time = 0
        
        # Calculate scenario difficulty for normalization
        self.scenario_difficulty = self._calculate_scenario_difficulty()
        
        self._update_state()
        
        return self.state, {"action_mask": self._get_action_mask()}
    
    def render(self, save_path=None):
        """Visualize the current route"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xlim([0, self.MAX_X])
        ax.set_ylim([0, self.MAX_Y])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Episode {self.episodes}: {self.customers_served}/{self.NUM_CUSTOMERS} served, {self.late_deliveries} late')
        ax.grid(True, alpha=0.3)
        
        # Draw depot
        ax.scatter(0, 0, color='green', s=400, marker='s', label='Depot', zorder=5)
        
        # Draw customers
        for i, cust in enumerate(self.customers):
            color = 'red' if cust['disrupted'] else 'blue'
            marker = 's' if cust['disrupted'] else 'o'
            ax.scatter(cust['x'], cust['y'], color=color, s=200, marker=marker, zorder=4)
            ax.annotate(f"{i+1}\nd={cust['deadline']}", 
                       xy=(cust['x'], cust['y']), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8)
        
        # Draw rejected customers
        for rej in self.rejected:
            ax.scatter(rej['x'], rej['y'], color='gray', s=100, marker='x', zorder=3)
        
        # Draw truck route
        truck_x, truck_y = 0, 0
        for dest in self.planned_route:
            if dest != 0:
                cust = self.customers[dest - 1]
                ax.plot([truck_x, cust['x']], [truck_y, cust['y']], 
                       'b-', linewidth=2, alpha=0.6)
                truck_x, truck_y = cust['x'], cust['y']
        
        # Draw drone routes
        drone_idx = 0
        truck_x, truck_y = 0, 0
        for dest in self.planned_route:
            if dest == 0 and drone_idx < len(self.drone_route):
                cust = self.customers[self.drone_route[drone_idx] - 1]
                ax.plot([truck_x, cust['x']], [truck_y, cust['y']], 
                       'r--', linewidth=2, alpha=0.6, label='Drone' if drone_idx == 0 else '')
                drone_idx += 1
            elif dest != 0:
                cust = self.customers[dest - 1]
                truck_x, truck_y = cust['x'], cust['y']
        
        ax.legend()
        
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
