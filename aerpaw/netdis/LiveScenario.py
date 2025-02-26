from random import random

class DisruptedScenario():
  def __init__(self):
    self.disruptions_enabled = True
    # self.random_disruptions = True
    self.requests = [
      {
        "x": 2,
        "y": 8,
        "deadline": 20,
        "disrupted": 0
      },
      {
        "x": 15,
        "y": 4,
        "deadline": 25,
        "disrupted": 0
      },
      {
        "x": 3,
        "y": 6,
        "deadline": 30,
        "disrupted": 0
      },
      {
        "x": 3,
        "y": 9,
        "deadline": 34,
        "disrupted": 0
      },
      # {
      #   "x": 19,
      #   "y": 18,
      #   "deadline": 40,
      #   "disrupted": 0
      # },
      {
        "x": 2,
        "y": 4,
        "deadline": 45,
        "disrupted": 0
      },
      {
        "x": 5,
        "y": 2,
        "deadline": 47,
        "disrupted": 0
      },
      {
        "x": 11,
        "y": 11,
        "deadline": 50,
        "disrupted": 0
      },
      {
        "x": 12,
        "y": 10,
        "deadline": 55,
        "disrupted": 0
      },
      {
        "x": 14,
        "y": 2,
        "deadline": 60,
        "disrupted": 0
      },
      {
        "x": 8,
        "y": 8,
        "deadline": 65,
        "disrupted": 0
      },
      {
        "x": 1,
        "y": 0,
        "deadline": 70,
        "disrupted": 0
      },
      {
        "x": 19,
        "y": 19,
        "deadline": 80,
        "disrupted": 0
      },
      {
        "x": 5,
        "y": 17,
        "deadline": 90,
        "disrupted": 0
      },
      {
        "x": 12,
        "y": 13,
        "deadline": 95,
        "disrupted": 0
      },
      {
        "x": 5,
        "y": 1,
        "deadline": 110,
        "disrupted": 0
      },
      {
        "x": 12,
        "y": 9,
        "deadline": 115,
        "disrupted": 0
      },
      {
        "x": 19,
        "y": 2,
        "deadline": 120,
        "disrupted": 0
      },
      {
        "x": 17,
        "y": 7,
        "deadline": 125,
        "disrupted": 0
      },
      {
        "x": 19,
        "y": 18,
        "deadline": 130,
        "disrupted": 0
      },
      {
        "x": 3,
        "y": 18,
        "deadline": 135,
        "disrupted": 0
      },
      {
        "x": 4,
        "y": 8,
        "deadline": 140,
        "disrupted": 0
      },
      {
        "x": 19,
        "y": 8,
        "deadline": 145,
        "disrupted": 0
      },
      {
        "x": 18,
        "y": 7,
        "deadline": 150,
        "disrupted": 0
      },
      {
        "x": 2,
        "y": 19,
        "deadline": 150,
        "disrupted": 0
      },
    ]


  def request(self):
    if len(self.requests) == 0:
      return None
    # print(self.requests[0])
    return self.requests.pop(0)

  def filter_requests(self, arr):
    valid_reqs = []
    for i in range(len(self.requests)):
      if (i+1) in arr:
        valid_reqs.append(self.requests[i])
    print('valid reqs')
    print(valid_reqs)
    self.requests = valid_reqs
  
