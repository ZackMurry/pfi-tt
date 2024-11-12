class TSPScenario():
  def __init__(self):
    self.requests = [
      {
        "x": 2,
        "y": 8,
        "deadline": 20
      },
      {
        "x": 15,
        "y": 4,
        "deadline": 25
      },
      {
        "x": 3,
        "y": 6,
        "deadline": 30
      },
      {
        "x": 3,
        "y": 9,
        "deadline": 34
      },
      {
        "x": 19,
        "y": 18,
        "deadline": 40
      },
      {
        "x": 2,
        "y": 4,
        "deadline": 45
      },
      {
        "x": 5,
        "y": 2,
        "deadline": 47
      },
      {
        "x": 11,
        "y": 11,
        "deadline": 50
      },
      {
        "x": 12,
        "y": 10,
        "deadline": 55
      },
      {
        "x": 14,
        "y": 2,
        "deadline": 60
      },
      {
        "x": 8,
        "y": 8,
        "deadline": 65
      },
      {
        "x": 1,
        "y": 0,
        "deadline": 70
      },
      {
        "x": 19,
        "y": 19,
        "deadline": 80
      },
      {
        "x": 5,
        "y": 17,
        "deadline": 90
      },
      {
        "x": 12,
        "y": 13,
        "deadline": 95
      },
      {
        "x": 5,
        "y": 1,
        "deadline": 110
      },
      {
        "x": 12,
        "y": 9,
        "deadline": 115
      },
      {
        "x": 19,
        "y": 2,
        "deadline": 120
      },
      {
        "x": 17,
        "y": 7,
        "deadline": 125
      },
      {
        "x": 19,
        "y": 18,
        "deadline": 130
      },
      {
        "x": 3,
        "y": 18,
        "deadline": 135
      },
      {
        "x": 4,
        "y": 8,
        "deadline": 140
      },
      {
        "x": 19,
        "y": 8,
        "deadline": 145
      },
      {
        "x": 18,
        "y": 7,
        "deadline": 150
      },
      {
        "x": 2,
        "y": 19,
        "deadline": 155
      },
    ]
    for req in self.requests:
      req['deadline'] *= 1


  def request(self):
    if len(self.requests) == 0:
      return None
    return self.requests.pop(0)
