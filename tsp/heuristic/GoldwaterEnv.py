import numpy as np
import gymnasium as gym
from gymnasium import spaces
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
from types import SimpleNamespace
from math import pow, sqrt

# todo: make sure drone returns before ending
class GoldwaterEnv(gym.Env):
    """
    RoutePEARL environment for training RL agents on network-aware
    truck-drone delivery route planning.
    """
    def __init__(self, num_customers=12, use_warm_start=True, test_mode=False, test_scenarios=None, *args, **kwargs):
        # Configuration
        self.NUM_CUSTOMERS = num_customers
        self.DRONE_SPEED_FACTOR = 1.5  # Drones are faster than trucks
        self.MAX_T = 240  # Maximum time horizon
        self.MAX_X = 20
        self.MAX_Y = 20
        self.DISRUPTION_PROB = 0.2  # 20% of customers have network disruptions
        self.use_warm_start = use_warm_start  # Whether to use heuristic initialization
        
        # Test mode: use fixed scenarios
        self.test_mode = test_mode
        self.test_scenarios = test_scenarios  # List of pre-generated scenarios
        self.test_scenario_idx = 0
        
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
        
        # Warm start state
        self.initial_plan = None  # Store the original heuristic plan
        self.disruptions_encountered = []  # Track which disruptions we've seen
        
        # Position tracking
        self.depot = {"x": 0, "y": 0}
        self.truck_x = 0
        self.truck_y = 0
        
        # Reward tracking - FIXED: Initialize served_customers properly
        self.served_customers = 0  # Customers successfully served (on-time, not rejected)
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
    
    def _manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points"""
        return abs(x2 - x1) + abs(y2 - y1)
    
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
                deadline = int(min_time * np.random.uniform(2.0, 4.5))
                
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
            deadline = 5 + int(min_time * np.random.uniform(2.0, 4.5))
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
    
    def _simulate_schedule(self):
        """
        Replays current planned_route and drone_route and returns:
        valid: bool
        makespan: float
        tardiness: float  (sum max(0, arrival - deadline) over served customers)
        """
        time = 0.0
        x, y = 0, 0
        tardiness = 0.0

        drone_with_truck = True
        drone_idx = 0
        drone_leg_time = 0.0
        drone_start_time = 0.0

        for dest in self.planned_route:
            if dest == 0:
                # Drone event toggles between "launch for current drone_idx" and "recover for current drone_idx"
                if drone_with_truck:
                    if drone_idx >= len(self.drone_route):
                        return False, 1e9, 1e9
                    cust_idx = self.drone_route[drone_idx] - 1
                    if cust_idx < 0 or cust_idx >= len(self.customers):
                        return False, 1e9, 1e9
                    cust = self.customers[cust_idx]

                    # Outbound leg from current truck position
                    drone_leg_time = float(self._get_travel_time(x, y, cust, is_drone=True))
                    drone_start_time = time

                    # Arrival time at customer (drone)
                    drone_arrival = drone_start_time + drone_leg_time
                    tardiness += max(0.0, drone_arrival - cust["deadline"])

                    drone_with_truck = False
                else:
                    # Recover drone: assume it returns from same customer to current truck position
                    cust_idx = self.drone_route[drone_idx] - 1
                    if cust_idx < 0 or cust_idx >= len(self.customers):
                        return False, 1e9, 1e9
                    cust = self.customers[cust_idx]

                    return_leg = float(self._get_travel_time(x, y, cust, is_drone=True))
                    total_drone = drone_leg_time + return_leg

                    elapsed_truck = time - drone_start_time
                    wait = max(0.0, total_drone - elapsed_truck)
                    time += wait

                    drone_idx += 1
                    drone_with_truck = True
                continue

            # Truck delivery
            if dest <= 0 or dest > len(self.customers):
                return False, 1e9, 1e9
            cust = self.customers[dest - 1]
            time += float(self._get_travel_time(x, y, cust, is_drone=False))
            tardiness += max(0.0, time - cust["deadline"])
            x, y = cust["x"], cust["y"]

        makespan = time
        return True, makespan, tardiness
    
    def _get_customer_position(self, customer_idx):
        """Get (x, y) position of a customer by index"""
        if customer_idx < 0 or customer_idx >= len(self.all_customers):
            return 0, 0  # Depot
        cust = self.all_customers[customer_idx]
        return cust['x'], cust['y']
    
    def _build_nearest_neighbor_route(self):
        """
        Build truck route using nearest-neighbor algorithm.
        Returns: (truck_route, rejected_customers)
        """
        truck_route = []
        remaining = set(range(len(self.all_customers)))
        
        current_x, current_y = 0, 0
        current_time = 0
        
        while remaining:
            # Find nearest feasible customer
            best_idx = None
            best_dist = float('inf')
            
            for idx in remaining:
                cust = self.all_customers[idx]
                dist = self._manhattan_distance(current_x, current_y, cust['x'], cust['y'])
                arrival_time = current_time + dist
                
                # Feasible if we can arrive before deadline
                if arrival_time <= cust['deadline'] and dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            
            if best_idx is None:
                break  # No more feasible customers
            
            truck_route.append(best_idx)
            remaining.remove(best_idx)
            
            current_x, current_y = self._get_customer_position(best_idx)
            current_time += best_dist
        
        return truck_route, list(remaining)
    
    def _calculate_route_makespan(self, truck_route, drone_sorties=None):
        """
        Calculate makespan for a given route configuration.
        
        Args:
            truck_route: List of customer indices for truck
            drone_sorties: List of (launch_idx, customer_idx, land_idx) tuples
                          or None for truck-only route
        
        Returns:
            Total makespan (time to complete all deliveries)
        """
        if drone_sorties is None:
            drone_sorties = []
        
        # Remove drone customers from truck route
        drone_customers = {sortie[1] for sortie in drone_sorties}
        truck_only = [idx for idx in truck_route if idx not in drone_customers]
        
        if not truck_only and not drone_sorties:
            return 0
        
        time = 0
        x, y = 0, 0
        
        # Create mapping of truck position -> sorties to launch
        sorties_by_launch = {}
        for sortie in drone_sorties:
            launch_idx = sortie[0]
            if launch_idx not in sorties_by_launch:
                sorties_by_launch[launch_idx] = []
            sorties_by_launch[launch_idx].append(sortie)
        
        # Track active drones: {sortie: completion_time}
        active_drones = {}
        
        # Simulate truck route
        for truck_idx, cust_idx in enumerate(truck_only):
            # Move truck to customer
            cust_x, cust_y = self._get_customer_position(cust_idx)
            time += self._manhattan_distance(x, y, cust_x, cust_y)
            x, y = cust_x, cust_y
            
            # Launch any drones scheduled at this position
            if truck_idx in sorties_by_launch:
                for sortie in sorties_by_launch[truck_idx]:
                    launch_idx, drone_cust_idx, land_idx = sortie
                    
                    # Calculate drone trip time
                    drone_cust_x, drone_cust_y = self._get_customer_position(drone_cust_idx)
                    
                    # Outbound: current position -> drone customer
                    drone_out = self._manhattan_distance(x, y, drone_cust_x, drone_cust_y) / self.DRONE_SPEED_FACTOR
                    
                    # Return: drone customer -> landing position
                    if land_idx >= len(truck_only):
                        land_x, land_y = self._get_customer_position(truck_only[-1]) if truck_only else (x, y)
                    else:
                        land_x, land_y = self._get_customer_position(truck_only[land_idx])
                    
                    drone_back = self._manhattan_distance(drone_cust_x, drone_cust_y, land_x, land_y) / self.DRONE_SPEED_FACTOR
                    
                    drone_completion = time + drone_out + drone_back
                    active_drones[sortie] = drone_completion
            
            # Wait for any drones landing at this position
            drones_to_remove = []
            for sortie, completion_time in active_drones.items():
                if sortie[2] == truck_idx:  # Drone lands here
                    time = max(time, completion_time)
                    drones_to_remove.append(sortie)
            
            for sortie in drones_to_remove:
                del active_drones[sortie]
        
        # Wait for any remaining drones
        if active_drones:
            time = max(time, max(active_drones.values()))
        
        return time
    
    def _is_sortie_feasible(self, truck_route, launch_idx, customer_idx, land_idx):
        """Check if a drone sortie is feasible"""
        if launch_idx >= land_idx or land_idx > len(truck_route):
            return False
        
        # Get positions
        launch_x, launch_y = self._get_customer_position(truck_route[launch_idx - 1]) if launch_idx > 0 else (0, 0)
        land_x, land_y = self._get_customer_position(truck_route[land_idx - 1]) if land_idx <= len(truck_route) else self._get_customer_position(truck_route[-1])
        drone_x, drone_y = self._get_customer_position(customer_idx)
        
        # Calculate times
        drone_time = (self._manhattan_distance(launch_x, launch_y, drone_x, drone_y) +
                     self._manhattan_distance(drone_x, drone_y, land_x, land_y)) / self.DRONE_SPEED_FACTOR
        
        # Calculate truck time for this segment
        truck_time = 0
        pos_x, pos_y = launch_x, launch_y
        for i in range(launch_idx, min(land_idx, len(truck_route))):
            next_x, next_y = self._get_customer_position(truck_route[i])
            truck_time += self._manhattan_distance(pos_x, pos_y, next_x, next_y)
            pos_x, pos_y = next_x, next_y
        
        # Feasible if drone completes before or shortly after truck arrives
        return drone_time <= truck_time + 10
    
    def _generate_heuristic_route(self):
        """
        Generate baseline route using nearest-neighbor heuristic.
        Uses truck only (no drones) to provide a simple, strong baseline.
        
        Algorithm:
        1. Start at depot
        2. Repeatedly visit nearest feasible customer
        3. Reject customers that cannot be reached on time
        
        Returns: (truck_route, drone_assignments, rejected_customers)
        """
        truck_route, rejected = self._build_nearest_neighbor_route()
        drone_assignments = []  # Heuristic doesn't use drones
        
        return truck_route, drone_assignments, rejected
    
    def _initialize_from_heuristic(self):
        """
        Initialize the episode with a heuristic plan.
        This represents the original route plan before disruptions were known.
        Builds planned_route where drone serves customers in parallel with truck.
        """
        truck_route, drone_assignments, rejected = self._generate_heuristic_route()
        
        # Store the initial plan for reference
        self.initial_plan = {
            'truck_route': truck_route.copy(),
            'drone_assignments': drone_assignments.copy(),
            'rejected': rejected.copy()
        }
        
        # Build planned_route with proper drone scheduling
        # Drone launches at one truck stop, serves customer, returns to later truck stop
        self.planned_route = []
        self.drone_route = []
        
        # Simple approach: interleave drone deliveries with truck stops
        # For each drone customer, insert: send drone (0), then later collect drone (0)
        
        truck_idx = 0
        drone_idx = 0
        
        # Add first few truck customers
        customers_before_first_drone = min(2, len(truck_route))
        for i in range(customers_before_first_drone):
            self.planned_route.append(truck_route[i] + 1)  # 1-indexed
            truck_idx += 1
        
        # Insert drone deliveries strategically
        while drone_idx < len(drone_assignments) and truck_idx < len(truck_route):
            # Launch drone
            self.planned_route.append(0)
            self.drone_route.append(drone_assignments[drone_idx] + 1)  # 1-indexed
            
            # Truck serves next 2-3 customers while drone is out
            customers_while_drone_out = min(2, len(truck_route) - truck_idx)
            for i in range(customers_while_drone_out):
                self.planned_route.append(truck_route[truck_idx] + 1)
                truck_idx += 1
            
            # Collect drone
            self.planned_route.append(0)
            
            drone_idx += 1
        
        # Add remaining truck customers
        while truck_idx < len(truck_route):
            self.planned_route.append(truck_route[truck_idx] + 1)
            truck_idx += 1
        
        # Add all customers to self.customers
        all_assigned = set(truck_route + drone_assignments)
        for idx in sorted(all_assigned):
            self.customers.append(self.all_customers[idx])
        
        # Rejected customers
        for idx in rejected:
            self.rejected.append(self.all_customers[idx])
        
        return truck_route, drone_assignments, rejected
    
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
            # FIXED: Calculate served_customers as customers delivered on time
            self.served_customers = len(self.customers) - num_late
            
            # Progressive reward for customers served
            reward += self.served_customers * 5  # Increased from 3
            
            # Big bonus for serving all customers on time
            if num_late == 0 and len(self.customers) == self.NUM_CUSTOMERS:
                reward += 20  # Increased from 10
            
            # Bonus for high service rate (based on on-time deliveries)
            service_rate = self.served_customers / self.NUM_CUSTOMERS
            if service_rate >= 0.9:
                reward += 10
            elif service_rate >= 0.75:
                reward += 5
            
            done = True
        
        self._update_state()
        
        return self.state, reward, done, False, {
            "action_mask": self._get_action_mask(),
            "served_customers": self.served_customers,  # On-time deliveries
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
        
        # Generate new customers (or use test scenario)
        if self.test_mode and self.test_scenarios:
            # Use fixed test scenario, cycle through them
            self.all_customers = self.test_scenarios[self.test_scenario_idx % len(self.test_scenarios)]
            self.test_scenario_idx += 1
        else:
            # Generate random customers for training
            self.all_customers = self._generate_customers()
        
        # Reset state
        self.customers = []
        self.planned_route = []
        self.drone_route = []
        self.rejected = []
        self.drone_with_truck = True
        self.truck_x = 0
        self.truck_y = 0
        self.disruptions_encountered = []
        
        # FIXED: Reset metrics including served_customers
        self.served_customers = 0
        self.late_deliveries = 0
        self.disrupted_drone_deliveries = 0
        self.previous_route_time = 0
        
        # Calculate scenario difficulty for normalization
        self.scenario_difficulty = self._calculate_scenario_difficulty()
        
        # WARM START: Initialize with heuristic route
        if self.use_warm_start:
            self._initialize_from_heuristic()
            # Now RL will process each customer sequentially and can modify the plan
            # Reset to process from scratch
            self.customers = []
            self.planned_route = []
            self.drone_route = []
            self.rejected = []
            self.drone_with_truck = True
        else:
            self.initial_plan = None
        
        self._update_state()
        
        return self.state, {"action_mask": self._get_action_mask()}
    
    @staticmethod
    def generate_test_scenarios(num_scenarios=25, num_customers=12, seed=42):
        """
        Generate a fixed set of test scenarios for consistent evaluation.
        
        Args:
            num_scenarios: Number of test scenarios to generate
            num_customers: Customers per scenario
            seed: Random seed for reproducibility
            
        Returns:
            List of customer lists (scenarios)
        """
        np.random.seed(seed)
        scenarios = []
        
        # Create environment temporarily to use its generation logic
        temp_env = GoldwaterEnv(num_customers=num_customers, test_mode=False)
        
        for i in range(num_scenarios):
            # Use different seed for each scenario
            np.random.seed(seed + i)
            customers = temp_env._generate_customers()
            scenarios.append(customers)
        
        return scenarios
    
    def render(self, save_path=None):
        """Visualize the current route and compare to heuristic baseline"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.set_xlim([0, self.MAX_X])
        ax.set_ylim([0, self.MAX_Y])
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Calculate final metrics
        final_valid, final_makespan, final_tardiness = self._simulate_schedule()
        title = f'Episode {self.episodes}: {self.served_customers}/{self.NUM_CUSTOMERS} served on-time'
        if self.initial_plan:
            initial_served = len(self.initial_plan['truck_route']) + len(self.initial_plan['drone_assignments'])
            improvement = self.served_customers - initial_served
            title += f' (Heuristic: {initial_served}, Δ={improvement:+d})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Draw depot
        ax.scatter(0, 0, color='green', s=500, marker='s', label='Depot', zorder=5, edgecolors='black', linewidths=2)
        
        # Draw heuristic route (if available) - in light gray/dashed
        if self.initial_plan:
            heuristic_x, heuristic_y = 0, 0
            
            # Draw heuristic truck route
            for cust_idx in self.initial_plan['truck_route']:
                cust = self.all_customers[cust_idx]
                ax.plot([heuristic_x, cust['x']], [heuristic_y, cust['y']], 
                       color='gray', linestyle='--', linewidth=1.5, alpha=0.5, 
                       label='Heuristic Route' if heuristic_x == 0 and heuristic_y == 0 else '')
                heuristic_x, heuristic_y = cust['x'], cust['y']
            
            # Draw heuristic drone assignments
            for cust_idx in self.initial_plan['drone_assignments']:
                cust = self.all_customers[cust_idx]
                ax.plot([0, cust['x']], [0, cust['y']], 
                       color='lightcoral', linestyle=':', linewidth=1.5, alpha=0.5,
                       label='Heuristic Drone' if cust_idx == self.initial_plan['drone_assignments'][0] else '')
        
        # Draw all customers (including rejected)
        for i, cust in enumerate(self.all_customers):
            if cust in self.rejected:
                # Rejected customers - gray X
                ax.scatter(cust['x'], cust['y'], color='gray', s=150, marker='x', 
                          zorder=3, linewidths=3)
                ax.annotate(f"REJ", xy=(cust['x'], cust['y']), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, color='gray', style='italic')
            elif cust in self.customers:
                # Served customers
                color = 'red' if cust['disrupted'] else 'blue'
                marker = 's' if cust['disrupted'] else 'o'
                ax.scatter(cust['x'], cust['y'], color=color, s=250, marker=marker, 
                          zorder=4, edgecolors='black', linewidths=1.5,
                          label='Disrupted' if cust['disrupted'] and i == 0 else ('Normal' if not cust['disrupted'] and i == 0 else ''))
                
                # Find this customer's index in self.customers
                cust_num = self.customers.index(cust) + 1
                ax.annotate(f"{cust_num}\nd={cust['deadline']}", 
                           xy=(cust['x'], cust['y']), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8, fontweight='bold')
            else:
                # Not processed yet (shouldn't happen in final render)
                ax.scatter(cust['x'], cust['y'], color='lightgray', s=100, marker='o', 
                          zorder=2, alpha=0.5)
        
        # Draw RL truck route (solid blue line)
        truck_x, truck_y = 0, 0
        truck_positions = [(0, 0)]
        for dest in self.planned_route:
            if dest != 0 and dest <= len(self.customers):
                cust = self.customers[dest - 1]
                truck_positions.append((cust['x'], cust['y']))
        
        if len(truck_positions) > 1:
            for i in range(len(truck_positions) - 1):
                ax.plot([truck_positions[i][0], truck_positions[i+1][0]], 
                       [truck_positions[i][1], truck_positions[i+1][1]], 
                       'b-', linewidth=3, alpha=0.8,
                       label='RL Truck Route' if i == 0 else '')
        
        # Draw RL drone routes (solid red dashed line)
        drone_idx = 0
        truck_x, truck_y = 0, 0
        drone_with_truck = True
        
        for dest in self.planned_route:
            if dest == 0:  # Drone action
                if drone_with_truck and drone_idx < len(self.drone_route):
                    # Send drone
                    cust = self.customers[self.drone_route[drone_idx] - 1]
                    ax.plot([truck_x, cust['x']], [truck_y, cust['y']], 
                           'r--', linewidth=3, alpha=0.8,
                           label='RL Drone Route' if drone_idx == 0 else '')
                    # Add arrow to show direction
                    dx = cust['x'] - truck_x
                    dy = cust['y'] - truck_y
                    ax.arrow(truck_x + 0.3*dx, truck_y + 0.3*dy, 
                            0.3*dx, 0.3*dy,
                            head_width=0.5, head_length=0.5, 
                            fc='red', ec='red', alpha=0.6)
                elif not drone_with_truck and drone_idx < len(self.drone_route):
                    # Collect drone
                    cust = self.customers[self.drone_route[drone_idx] - 1]
                    ax.plot([truck_x, cust['x']], [truck_y, cust['y']], 
                           'g:', linewidth=2, alpha=0.6)  # Return path (dotted)
                    drone_idx += 1
                drone_with_truck = not drone_with_truck
            elif dest > 0 and dest <= len(self.customers):
                cust = self.customers[dest - 1]
                truck_x, truck_y = cust['x'], cust['y']
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add text box with statistics - FIXED: Use served_customers correctly
        stats_text = f"RL Performance:\n"
        stats_text += f"  On-time: {self.served_customers}/{self.NUM_CUSTOMERS}\n"
        stats_text += f"  Late: {self.late_deliveries}\n"
        stats_text += f"  Rejected: {len(self.rejected)}\n"
        stats_text += f"  Drone Uses: {len(self.drone_route)}\n"
        if final_valid:
            stats_text += f"  Makespan: {final_makespan:.1f}\n"
            stats_text += f"  Tardiness: {final_tardiness:.1f}"
        
        if self.initial_plan:
            stats_text += f"\n\nHeuristic Baseline:\n"
            stats_text += f"  Served: {initial_served}/{self.NUM_CUSTOMERS}\n"
            stats_text += f"  Rejected: {len(self.initial_plan['rejected'])}\n"
            stats_text += f"  Drone Uses: {len(self.initial_plan['drone_assignments'])}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
