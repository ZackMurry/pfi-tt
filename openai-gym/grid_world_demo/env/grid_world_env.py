import gym
from gym.utils import seeding
from gym import spaces
import pygame
import numpy as np
from math import sqrt

SQRT2 = sqrt(2)

class GridWorldEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}
  _np_random = None

  def __init__(self, render_mode=None, size=5):
    self.size = size # Size of grid
    self.window_size = 512

    self.observation_space = spaces.Dict(
      {
        "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        "target_rel": spaces.Box(0, size - 1, shape=(2,), dtype=int)
      }
    )

    self.action_space = spaces.Discrete(4) # RULD

    self._action_to_direction = {
      0: np.array([1,0]),
      1: np.array([0,1]),
      2: np.array([-1,0]),
      3: np.array([0,-1])
    }

    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

    self.window = None
    self.clock = None

  def _get_obs(self):
    return {"agent": self._agent_location, "target_rel": self._target_location - self._agent_location}

  def _get_info(self):
    return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

  def reset(self, seed=None, options=None):
    super()
    if self._np_random is None:
      seed_seq = np.random.SeedSequence(seed)
      rng = np.random.Generator(np.random.PCG64(seed_seq))
      self._np_random = rng
    self._agent_location = self._np_random.integers(0, self.size, size=2, dtype=int)
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
      self._target_location = self._np_random.integers(
        0, self.size, size=2, dtype=int
      )
    
    observation = self._get_obs()
    info = self._get_info()
    
    if self.render_mode == "human":
      self._render_frame()
    
    return observation, info

  def step(self, action):
    direction = self._action_to_direction[action]
    self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)
    terminated = np.array_equal(self._agent_location, self._target_location)
    info = self._get_info()
    reward = 1 if terminated else .5 - (info["distance"] / (2*self.size*SQRT2))
    # reward = 1 if terminated else 0
    observation = self._get_obs()
    # print(f"agent: {self._agent_location}, target: {self._target_location}, delta: {observation['target_rel']}")

    if self.render_mode == "human":
      self._render_frame()
    
    return observation, reward, terminated, False, info

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()

  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode((self.window_size, self.window_size))
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()
    
    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255,255,255))
    pix_square_size = self.window_size / self.size
    # Draw the target
    pygame.draw.rect(
      canvas,
      (255,0,0),
      pygame.Rect(
        pix_square_size * self._target_location, (pix_square_size, pix_square_size)
      )
    )

    # Draw the agent
    pygame.draw.circle(
      canvas,
      (0,0,255),
      (self._agent_location + 0.5) * pix_square_size,
      pix_square_size/3
    )

    # Draw gridlines
    for i in range(self.size + 1):
      pygame.draw.line(
        canvas,
        0,
        (0, pix_square_size * i),
        (self.window_size, pix_square_size * i),
        width=3
      )
      pygame.draw.line(
        canvas,
        0,
        (pix_square_size * i, 0),
        (pix_square_size * i, self.window_size),
        width=3
      )
    
    if self.render_mode == "human":
      # Copy canvas to window
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()
      self.clock.tick(self.metadata["render_fps"])
    else:
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
      )

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()
