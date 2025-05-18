from random import random

class LiveScenario():
  def __init__(self):
    self.disruptions_enabled = True
    self.order_idx = 0
    self.order = []
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
    if self.order_idx >= len(self.order):
      print('No more requests!')
      return None
    req = self.requests[self.order[self.order_idx]]
    self.order_idx += 1
    print('LIVE SCENARIO: ', req, ' requested')
    return req

  def reset(self):
    self.order_idx = 0

  def set_served_custs(self, the_list):
    self.order = the_list
  
  # When we give a preset planned route, we need to skip to idx to not re-give the preset customers
  def set_order_index(self, idx):
    self.order_idx = idx

  def get_served_custs(self):
    custs = []
    for idx in self.order:
      custs.append(self.requests[idx-1])
    print('Customers in order', custs)
    return custs

  # Gets served custs that appear in the list
  def get_served_custs(self, the_list):
    custs = []
    for idx in the_list:
      custs.append(self.requests[idx-1])
    print('Served custs in order', custs)
    return custs

  
  # Switch from absolute indices to served indices
  # ex for self.order = [1,3,4,5,6,9,7,8]:
  # [0,3,0,4,0,6,9,7,8,0] -> [0,2,0,3,0,5,6,7,8,0]
  def translate_custs(self, cust_list):
    custs = []
    print('Translating cust_list', cust_list)
    for c in cust_list:
      if c != 0:
        custs.append(self.order.index(c)+1)
      else:
        custs.append(0)
    print('Translated', custs)
    return custs

  def untranslate_custs(self, cust_list):
    custs = []
    print('Untranslating cust_list', cust_list)
    print('Order', self.order)
    for c in cust_list:
      if c != 0:
        print("Finding original value for index ", c-1)
        custs.append(self.order[c-1])
      else:
        custs.append(0)
    print('Untranslated', custs)
    return custs


