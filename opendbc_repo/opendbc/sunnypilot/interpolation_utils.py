import numpy as np


def catmull_rom_interp(x, xp, fp, alpha=0.5):
  """1D Catmull-Rom interpolation for vehicle dynamics.

  Args:
    x: The input value to interpolate at
    xp: Array of x-coordinates of points
    fp: Array of y-coordinates of points
    alpha: Parameter affecting tension (0.5=centripetal recommended)

  Returns:
    Interpolated value at x
  """
  # Handle boundary conditions
  if x <= xp[0]:
    return fp[0]
  elif x >= xp[-1]:
    return fp[-1]

  # Find the segment containing x
  i = np.searchsorted(xp, x) - 1
  i = max(0, min(i, len(xp)-2))

  # Get the 4 control points needed
  if i == 0:
    p0 = 2*fp[0] - fp[1]  # Extrapolate
  else:
    p0 = fp[i-1]

  p1 = fp[i]
  p2 = fp[i+1]

  if i+2 >= len(fp):
    p3 = 2*fp[-1] - fp[-2]  # Extrapolate
  else:
    p3 = fp[i+2]

  # Calculate parameter values using the chosen alpha
  t0 = 0.0
  t1 = t0 + abs(p1 - p0)**alpha
  t2 = t1 + abs(p2 - p1)**alpha
  t3 = t2 + abs(p3 - p2)**alpha
  t = t1 + (t2 - t1) * (x - xp[i]) / (xp[i+1] - xp[i])

  # Catmull-Rom basis functions
  s = (t - t1) / (t2 - t1)

  # Hermite basis functions
  h00 = 2*s**3 - 3*s**2 + 1
  h10 = s**3 - 2*s**2 + s
  h01 = -2*s**3 + 3*s**2
  h11 = s**3 - s**2

  # Calculate tangents
  m1 = (p2 - p0) / (t2 - t0) * (t2 - t1)
  m2 = (p3 - p1) / (t3 - t1) * (t2 - t1)

  return h00*p1 + h10*m1 + h01*p2 + h11*m2

def makima_interp(x, xp, fp):
  """Modified Akima interpolation similar to MATLAB's implementation."""
  # Handle boundary conditions
  if x <= xp[0]:
    return fp[0]
  elif x >= xp[-1]:
    return fp[-1]

  # Find the interval and calculate slopes
  i = np.searchsorted(xp, x) - 1
  i = max(0, min(i, len(xp)-2))
  n = len(xp)
  slopes = np.zeros(n)

  for j in range(1, n-1):
    dx1, dx2 = xp[j] - xp[j-1], xp[j+1] - xp[j]
    dy1, dy2 = fp[j] - fp[j-1], fp[j+1] - fp[j]

    # Calculate finite differences with modified weights
    s1 = dy1 / max(dx1, 1e-6)
    s2 = dy2 / max(dx2, 1e-6)
    w1 = abs(s1) * dx2
    w2 = abs(s2) * dx1

    # Weighted average
    if (s1 * s2) > 0:  # Same sign -> preserve monotonicity
      slopes[j] = (w1 * s2 + w2 * s1) / max(w1 + w2, 1e-6)
    else:  # Different signs or zero -> reduce oscillation
      slopes[j] = 0

  # Endpoint conditions
  slopes[0] = 2.0 * (fp[1] - fp[0]) / max(xp[1] - xp[0], 1e-6) - slopes[1]
  slopes[-1] = 2.0 * (fp[-1] - fp[-2]) / max(xp[-1] - xp[-2], 1e-6) - slopes[-2]
  slopes[0] *= 0.5
  slopes[-1] *= 0.5

  # Interpolate using Hermite cubic formula
  h = xp[i+1] - xp[i]
  t = (x - xp[i]) / h
  h00 = 1 - 3*t*t + 2*t*t*t
  h10 = t - 2*t*t + t*t*t
  h01 = 3*t*t - 2*t*t*t
  h11 = -t*t + t*t*t

  return fp[i] * h00 + h * slopes[i] * h10 + fp[i+1] * h01 + h * slopes[i+1] * h11

