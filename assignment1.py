import numpy as np
from sim.sim1d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['CONSTANT_SPEED'] = False

class KalmanFilterToy:
    def __init__(self):
      self.v = 0
      self.prev_x = 0
      self.prev_t = 0
    def predict(self,t):
      # returns the predicted value of x
      dt = t - self.prev_t
      # prediction = 0
      prediction = self.prev_x + (self.v * dt)
      return prediction
    def measure_and_update(self,x,t):
      # update the value of v
      measured_v = (x - self.prev_x) / (t - self.prev_t)
      self.v += 0.65 * (measured_v - self.v)

      self.prev_x = x
      self.prev_t = t
      return


sim_run(options,KalmanFilterToy)
