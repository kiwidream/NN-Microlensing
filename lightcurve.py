import numpy as np
import random
import time
import statistics
from raw_nodes import excursion, pspec
from scipy.ndimage import gaussian_filter

class LightCurve:

  CURVE_X = 0
  CURVE_Y = 1
  CURVE_SIGMA = 2
  CURVE_X_CLEAN = 3
  CURVE_Y_CLEAN = 4

  # This varies the smoothing accuracy when using gaussian_filter
  SMOOTH_SIGMA = 2

  INPUT_SIZE = 9
  OUTPUT_SIZE = 3

  def __init__(self):
    self.input_neurons = []
    self.params = self.params or []
    self.curve = None
    self.size = 600
    self.corr = [None, None]
    self.smoothed = None
    self.excurs = None
    self.generate_params()

  def generate_params(self):
    """ Overridden by child classes """
    pass

  def generate_curve(self):
    """ Overridden by child classes """
    pass

  def calculate_inputs(self):
    if self.curve is None:
      self.generate_curve()

    input_list = [self.ac_width, self.ac_max, self.ac_symm_width, self.ac_symm_max, self.excursion_above, self.excursion_below, self.noise_est, self.fitted_slope, self.power_spec]
    inputs = np.zeros((self.INPUT_SIZE, 1))
    for i in range(self.INPUT_SIZE):
      inputs[i] = input_list[i]()
    return inputs

  def ac_width(self):
    return statistics.stdev(self.autocorrelate())

  def ac_max(self):
    return max(self.autocorrelate())

  def ac_symm_max(self):
    return max(self.autocorrelate(True))

  def ac_symm_width(self):
    return statistics.stdev(self.autocorrelate(True))

  def excursion_above(self):
    if self.excurs is not None:
      return self.excurs[0]

    self.excurs = excursion(self.curve[:, :2])
    return self.excurs[0]

  def excursion_below(self):
    if self.excurs is not None:
      return self.excurs[1]

    self.excurs = excursion(self.curve[:, :2])
    return self.excurs[1]

  def noise_est(self):
    return np.mean(self.curve[:, self.CURVE_SIGMA]) / statistics.stdev(self.curve[:, self.CURVE_Y])

  def fitted_slope(self):
    return np.polyfit(self.curve[:, self.CURVE_X], self.curve[:, self.CURVE_Y], 1, w=(1/self.curve[:, self.CURVE_SIGMA]))[0]

  def power_spec(self):
    return pspec(self)

  def autocorrelate(self, rev=False):
    if self.corr[int(rev)] is not None:
      return self.corr[int(rev)]

    t, y = self.interpolate_smooth()
    y2 = np.flip(y, axis=0) if rev else y
    corr = np.correlate(y, y2, mode='same')
    n = len(corr)

    lengths = range(n, n//2, -1)
    ac = corr[n//2:] / lengths

    self.corr[int(rev)] = ac / ac[0] # Normalise

    return self.corr[int(rev)]

  def interpolate_smooth(self):
    if self.smoothed is not None:
      return self.smoothed

    t = np.linspace(self.curve[0, self.CURVE_X], self.curve[-1, self.CURVE_X], self.size * 5)
    interpolated = np.interp(t, self.curve[:, self.CURVE_X], self.curve[:, self.CURVE_Y])

    self.smoothed = t, gaussian_filter(interpolated, self.SMOOTH_SIGMA)
    return self.smoothed

  def expected_outputs(self):
    light_curves = [NonEvent, MicroLensing, Periodic]
    outputs = np.zeros((self.OUTPUT_SIZE, 1))
    for i in range(self.OUTPUT_SIZE):
      outputs[i, 0] = int(isinstance(self, light_curves[i]))
    return outputs

  def get_filters(self):
    return [self.noise_sigma_filter, self.patchy_filter]

  def apply_filters(self):
    self.curve[:, self.CURVE_X_CLEAN] = self.curve[:, self.CURVE_X]
    self.curve[:, self.CURVE_Y_CLEAN] = self.curve[:, self.CURVE_Y]

    for filter in self.get_filters():
      filter()

    return self.curve

  def noise_sigma_filter(self):
    n = 10 ** random.randint(-1, 1)
    errbar = np.abs(np.random.normal(0, n, self.size))
    noise = np.random.normal(0, errbar)
    self.curve[:, self.CURVE_SIGMA] = errbar
    self.curve[:, self.CURVE_Y] = self.curve[:, self.CURVE_Y] + noise

  def patchy_filter(self):
    remove = []
    skip = 0
    for i in range(self.size):
      if skip > 0:
        skip -= 1
      if random.randint(0, 1) == 0 or skip > 0:
        remove.append(i)
        continue
      if random.randint(0, 20) == 0:
        skip = random.randint(0, 8)

    self.curve = np.delete(self.curve, remove, axis=0)
    self.size -= len(remove)

class NonEvent(LightCurve):
  def __init__(self, m=None, c=None):
    self.params = [m, c]
    self.name = 'Non-Event Curve'
    super().__init__()

  def generate_params(self):
    if len(self.params) == 2:
      self.m, self.c = self.params

    self.m = self.m or random.uniform(-0.9999999, 1.0000001)
    self.c = self.c or random.uniform(20, 100)

  def generate_curve(self):
    self.curve = np.zeros((self.size, 5))
    self.curve[:, self.CURVE_X] = np.linspace(0, self.size, self.size)
    self.curve[:, self.CURVE_Y] = self.curve[:, self.CURVE_X] * self.m + self.c

    return self.apply_filters()

class MicroLensing(LightCurve):

  def __init__(self, uo=None, tE=None, to=None):
    self.params = [uo, tE, to]
    self.name = 'MicroLensing Event Curve'
    super().__init__()

  def generate_params(self):
    if len(self.params) == 3:
      self.uo, self.tE, self.to = self.params

    self.uo = self.uo or random.uniform(0, 1.5)
    self.tE = self.tE or random.uniform(2, 30)
    self.to = self.to or random.uniform(100, 5000)

  def generate_curve(self):
    shift = random.uniform(-200, 200)
    self.curve = np.zeros((self.size, 5))
    self.curve[:, self.CURVE_X] = np.linspace(self.to - self.size/2 + shift, self.to + self.size/2 + shift, self.size)
    self.curve[:, self.CURVE_Y] = self.total_magnification(self.rel_lense_motion(self.uo, self.curve[:, self.CURVE_X], self.tE, self.to))

    self.apply_filters()

    return self.curve


  @staticmethod
  def total_magnification(u):
    """ Calculates the total magnification 'a' for a given 'u' value. """
    return ((u**2) + 2) / (u * ((u**2) + 4)**(1/2))

  @staticmethod
  def rel_lense_motion(uo, t, tE, to):
    """ Calculates the relative lens motion 'u' from the time and closest separation
    parameters. """
    return ( (uo**2) + (((t - to) / tE)**2) )** (1/2)

class Periodic(LightCurve):

  def __init__(self, skew=None, amp=None, subAmp=None, subFreq=None, mean=None, period=None):
    self.params = [skew, amp, subAmp, subFreq, mean, period]
    self.name = 'Periodic Curve'
    super().__init__()

  def generate_params(self):
    if len(self.params) == 6:
      self.skew, self.amp, self.subAmp, self.subFreq, self.mean, self.period = self.params

    self.skew = self.skew or 1 / random.uniform(1, 10)
    self.amp = self.amp or random.uniform(1, 30)
    self.subAmp = self.subAmp or self.amp / random.uniform(3, 15)
    self.subFreq = self.subFreq or random.uniform(10, 15)
    self.mean = self.mean or self.amp + random.uniform(150, 1000)
    self.period = self.period or random.uniform(10, 100)

  def generate_curve(self):
    phase = random.uniform(0, 700)
    self.curve = np.zeros((self.size, 5))
    self.curve[:, self.CURVE_X] = np.linspace(phase, self.size + phase, self.size)
    t = self.curve[:, self.CURVE_X]
    self.curve[:, self.CURVE_Y] = self.mean + self.amp * np.sin(self.curve[:, self.CURVE_X] / self.period + np.sin(self.skew * self.curve[:, self.CURVE_X] / self.period)) + self.subAmp * np.sin(self.subFreq * self.curve[:, self.CURVE_X] / self.period)

    self.apply_filters()

    return self.curve

if __name__ == '__main__':
  ml = MicroLensing()
  print(ml.calculate_inputs())