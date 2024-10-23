# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:09:05 2024

@author: bvlsc
"""

import numpy as np
import random
import pygame
import sys
import time
import math
import heapq

# Initialize Pygame
pygame.init()

# Set the dimensions of the window
width, height = 800, 800
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Truck and Points Visualization with Q-learning")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

cheese_eaten = False
vx, vy = 0, 0

# Define obstacles as rectangles
obstacles = [
    pygame.Rect(370, 300, 400, 100),
    pygame.Rect(175, 200, 100, 200),
    pygame.Rect(450, 650, 100, 100)
]

congestion = [
    pygame.Rect(75, 75, 100, 100)
]
# Generate random points within the window bounds
def generate_random_point():
    random.seed(128)
    return (random.randint(100, width-100), random.randint(100, height-100))
    # return (width // 2, height // 2)


# Function to check if within radius
def is_within_radius(point1, point2, radius=20):
    return np.hypot(point1[0] - point2[0], point1[1] - point2[1]) <= radius

def distance(point1, point2):
    return np.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)

# Function to get reward
def get_reward(drone, customer, emergency, depot, delivered, truck, speed):
    for obstacle in obstacles:
            if obstacle.collidepoint(drone):
                # print("collide bad")
                return -1_000_000_000_000
    if drone[0] == width - 1 or drone[0] == 0 or drone[1] == height or drone[1] == 0:
        return -1_000_000_000_000
    reward = 0
    for cong in congestion: 
        if cong.collidepoint(drone):
            reward -= 1000
    
    reward -= (distance(drone, truck)/100 - 0.1*(abs(vx) + abs(vy)))

    if not delivered:
        if is_within_radius(drone, customer):
            return reward + 5000
    else:    
        if is_within_radius(drone, depot):
            return reward + 25
        if is_within_radius(drone, emergency):
            return reward + 5000
    
        
    return reward - 10 - 0.1*(abs(vx) + abs(vy))  # Penalty proportional to velocity
    


# Font for displaying text
font = pygame.font.Font(None, 36)
delivered_rewards = []
path = set()
drone = (100, 100)
customer = (100, 700)
depot = (700, 700)
emergency = (700, 150)
truck = generate_random_point()
# Training the Q-learning agent with visualization
window.fill(WHITE)

pygame.draw.circle(window, RED, drone, 5)

delivered = False
target = None
reward = 0

class Cell:
    def __init__(self):
        self.parent_i = 0  # Parent cell's row index
        self.parent_j = 0  # Parent cell's column index
        self.f = float('inf')  # Total cost of the cell (g + h)
        self.g = float('inf')  # Cost from start to this cell
        self.h = 0  # Heuristic cost from this cell to destination

def calculate_h_value(x, y, dest):
    return ((x - dest[0]) ** 2 + (y - dest[1]) ** 2) ** 0.5
    # return min(abs(dest[0]-col), abs(dest[1]-row))

def is_valid(x, y):
    return (x >= 0) and (x < width) and (y >= 0) and (y < height)

def is_unblocked(x, y):
    for obst in obstacles:
        if obst.collidepoint(x, y):
            return False
    return True


# Check if a cell is the destination
def is_destination(x, y, dest):
    return x == dest[0] and y == dest[1]

first_delivered = False
total_reward = 0

# Trace the path from source to destination
def trace_path(cells, dest):
    print("The Path is ")
    path = []
    row = dest[0]
    col = dest[1]

    # Trace the path from destination to source using parent cells
    while not (cells[row][col].parent_i == row and cells[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cells[row][col].parent_i
        temp_col = cells[row][col].parent_j
        row = temp_row
        col = temp_col

    # Add the source cell to the path
    path.append((row, col))
    # Reverse the path to get the path from source to destination
    path.reverse()

    # Print the path
    global total_reward
    path_reward = 0
    for i in path:
        print("->", i, end=" ")
        pygame.draw.rect(window, CYAN, pygame.Rect(i[0], i[1], 5, 5))
        reward = get_reward(i, customer, emergency, depot, first_delivered, truck, 1)
        print(f"Reward: {reward}")
        path_reward += reward
        sleep(0.05)
    
    total_reward += path_reward
    print(f"Path reward: {path_reward}")
    print(f"Total reward: {total_reward}")
    print()
    pygame.draw.rect(window, RED, pygame.Rect(200, 100, 5, 5))


# A* to customer
def a_star(src, target):
    print(f"A* from {src} to {target}")
    closed_list = [[False for _ in range(width)] for _ in range(height)]
    cells = [[Cell() for _ in range(width)] for _ in range(height)]
    i = src[0]
    j = src[1]
    print(f"Starting at {i}, {j}")
    cells[i][j].f = 0
    cells[i][j].g = 0
    cells[i][j].h = 0
    cells[i][j].parent_i = i
    cells[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))
    found_dest = False

    while len(open_list) > 0:
        p = heapq.heappop(open_list)
        i = p[1]
        j = p[2]
        closed_list[i][j] = True
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in dirs:
            new_i = i + dir[0]
            new_j = j + dir[1]
            # print(f"Testing {new_i}, {new_j}")
            if is_valid(new_i, new_j) and is_unblocked(new_i, new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i, new_j, target):
                    # Set the parent of the destination cell
                    cells[new_i][new_j].parent_i = i
                    cells[new_i][new_j].parent_j = j
                    print(f"Parent: {cells[new_i][new_j].parent_i}, {cells[new_i][new_j].parent_j}")
                    print("The destination cell is found")
                    # Trace and print the path from source to destination
                    trace_path(cells, target)
                    found_dest = True
                    break
                else:
                    g_new = cells[i][j].g + 1.0
                    h_new = calculate_h_value(new_i, new_j, target)
                    f_new = g_new + h_new
                    # If the cell is not in the open list or the new f value is smaller
                    if cells[new_i][new_j].f == float('inf') or cells[new_i][new_j].f > f_new:
                        # Add the cell to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the cell details
                        cells[new_i][new_j].f = f_new
                        cells[new_i][new_j].g = g_new
                        cells[new_i][new_j].h = h_new
                        cells[new_i][new_j].parent_i = i
                        cells[new_i][new_j].parent_j = j
        if found_dest:
            break

src = (100,700) #drone
target = (700, 150) #customer

a_star(drone, customer)
first_delivered = True
a_star(customer, depot)

for i in range(100):
    if not delivered:
        pygame.draw.circle(window, GREEN, customer, 20)
    pygame.draw.circle(window, BLUE, depot, 20)
    pygame.draw.circle(window, CYAN, emergency, 20)
    for cong in congestion:
        pygame.draw.rect(window, YELLOW, cong)
    for obstacle in obstacles:
        pygame.draw.rect(window, ORANGE, obstacle)
    for p in path:
        pygame.draw.circle(window, (0, 255, 0), p, 5)
    # Draw obstacles




    if target is not None:
        pygame.draw.line(window, GREEN, drone, customer, 1)
        pygame.draw.line(window, BLUE, drone, depot, 1)
        pygame.draw.line(window, CYAN, drone, emergency, 1)
    pygame.draw.circle(window, BLACK, truck, 5)

    # vx_text = font.render(f'Last reward: {reward}', True, BLACK)
    vy_text = font.render(f'Total reward: {total_reward:.2f}', True, BLACK)
    # window.blit(vx_text, (10, 10))
    window.blit(vy_text, (10, 10))

    pygame.display.flip()

    time.sleep(0.01)

print("final path")
for p in path:
    print(p)

time.sleep(120)

pygame.quit()
sys.exit()
