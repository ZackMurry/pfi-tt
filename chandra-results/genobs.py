import sys
import random
def generate_random_coordinates(N):
    # Generate coordinates in the range [0, 14], excluding (0, 0) and (14, 14)
    all_coords = [(x, y) for x in range(15) for y in range(15) if (x, y) not in [(0, 0), (14, 14)]]
    if N > len(all_coords):
      raise ValueError(f"Cannot generate {N} unique coordinates from the available {len(all_coords)} points.")
    return random.sample(all_coords, N)

print(generate_random_coordinates(int(sys.argv[1])))

