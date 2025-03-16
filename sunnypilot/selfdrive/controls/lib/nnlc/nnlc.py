"""
The MIT License

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Last updated: July 29, 2024
"""
from collections import deque
import math
import numpy as np

from opendbc.car.interfaces import LatControlInputs
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants

LOW_SPEED_Y_NN = [12, 3, 1, 0]
LAT_PLAN_MIN_IDX = 5


def get_predicted_lateral_jerk(lat_accels, t_diffs):
  # compute finite difference between subsequent model_v2.acceleration.y values
  # this is just two calls of np.diff followed by an element-wise division
  lat_accel_diffs = np.diff(lat_accels)
  lat_jerk = lat_accel_diffs / t_diffs
  # return as python list
  return lat_jerk.tolist()


def sign(x):
  return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def get_lookahead_value(future_vals, current_val):
  if len(future_vals) == 0:
    return current_val

  same_sign_vals = [v for v in future_vals if sign(v) == sign(current_val)]

  # if any future val has opposite sign of current val, return 0
  if len(same_sign_vals) < len(future_vals):
    return 0.0

  # otherwise return the value with minimum absolute value
  min_val = min(same_sign_vals + [current_val], key=lambda x: abs(x))
  return min_val


# At a given roll, if pitch magnitude increases, the
# gravitational acceleration component starts pointing
# in the longitudinal direction, decreasing the lateral
# acceleration component. Here we do the same thing
# to the roll value itself, then passed to nnff.
def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl:
  def __init__(self, lac_torque):
    self.CP = lac_torque.CP
    self.CI = lac_torque.CI

    self.torque_from_lateral_accel = lac_torque.torque_from_lateral_accel
    self.torque_params = lac_torque.torque_params

    self.model_v2 = None
    self.model_valid = False
    self.use_lateral_jerk: bool = False  # TODO: make this a parameter in the UI
    self.use_steering_angle = lac_torque.use_steering_angle

    self.actual_lateral_jerk: float = 0.0
    self.lateral_jerk_setpoint: float = 0.0
    self.lateral_jerk_measurement: float = 0.0
    self.lookahead_lateral_jerk: float = 0.0

    # twilsonco's Lateral Neural Network Feedforward
    self.use_nn = self.CI.has_lateral_torque_nn  # FIXME-SP: cereal exists

    if self.use_nn or self.use_lateral_jerk:
      # Instantaneous lateral jerk changes very rapidly, making it not useful on its own,
      # however, we can "look ahead" to the future planned lateral jerk in order to guage
      # whether the current desired lateral jerk will persist into the future, i.e.
      # whether it's "deliberate" or not. This lets us simply ignore short-lived jerk.
      # Note that LAT_PLAN_MIN_IDX is defined above and is used in order to prevent
      # using a "future" value that is actually planned to occur before the "current" desired
      # value, which is offset by the steerActuatorDelay.
      self.friction_look_ahead_v = [1.4, 2.0] # how many seconds in the future to look ahead in [0, ~2.1] in 0.1 increments
      self.friction_look_ahead_bp = [9.0, 30.0] # corresponding speeds in m/s in [0, ~40] in 1.0 increments

      # Scaling the lateral acceleration "friction response" could be helpful for some.
      # Increase for a stronger response, decrease for a weaker response.
      self.lat_jerk_friction_factor = 0.4
      self.lat_accel_friction_factor = 0.7 # in [0, 3], in 0.05 increments. 3 is arbitrary safety limit

      # precompute time differences between ModelConstants.T_IDXS
      self.t_diffs = np.diff(ModelConstants.T_IDXS)
      self.desired_lat_jerk_time = self.CP.steerActuatorDelay + 0.3
    if self.use_nn:
      self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
      # NN model takes current v_ego, lateral_accel, lat accel/jerk error, roll, and past/future/planned data
      # of lat accel and roll
      # Past value is computed using previous desired lat accel and observed roll
      self.torque_from_nn = self.CI.get_ff_nn  # FIXME-SP: cereal exists
      self.nn_friction_override = self.CI.lat_torque_nn_model.friction_override  # FIXME-SP: cereal exists

      # setup future time offsets
      self.nn_time_offset = self.CP.steerActuatorDelay + 0.2
      future_times = [0.3, 0.6, 1.0, 1.5] # seconds in the future
      self.nn_future_times = [i + self.nn_time_offset for i in future_times]
      self.nn_future_times_np = np.array(self.nn_future_times)

      # setup past time offsets
      self.past_times = [-0.3, -0.2, -0.1]
      history_check_frames = [int(abs(i)*100) for i in self.past_times]
      self.history_frame_offsets = [history_check_frames[0] - i for i in history_check_frames]
      self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
      self.roll_deque = deque(maxlen=history_check_frames[0])
      self.error_deque = deque(maxlen=history_check_frames[0])
      self.past_future_len = len(self.past_times) + len(self.nn_future_times)

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.orientation.x) >= CONTROL_N

  def update_lateral_jerk(self, CS, VM, desired_lateral_accel):
    self.actual_lateral_jerk = 0.0
    self.lateral_jerk_setpoint = 0.0
    self.lateral_jerk_measurement = 0.0
    self.lookahead_lateral_jerk = 0.0

    if self.use_steering_angle:
      if self.use_nn or self.use_lateral_jerk:
        actual_curvature_rate = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0)
        self.actual_lateral_jerk = actual_curvature_rate * CS.vEgo ** 2

    if self.model_valid and (self.use_nn or self.use_lateral_jerk):
      # prepare "look-ahead" desired lateral jerk
      lookahead = np.interp(CS.vEgo, self.friction_look_ahead_bp, self.friction_look_ahead_v)
      friction_upper_idx = next((i for i, val in enumerate(ModelConstants.T_IDXS) if val > lookahead), 16)
      predicted_lateral_jerk = get_predicted_lateral_jerk(self.model_v2.acceleration.y, self.t_diffs)
      desired_lateral_jerk = (np.interp(self.desired_lat_jerk_time, ModelConstants.T_IDXS,
                                     self.model_v2.acceleration.y) - desired_lateral_accel) / self.desired_lat_jerk_time
      self.lookahead_lateral_jerk = get_lookahead_value(predicted_lateral_jerk[LAT_PLAN_MIN_IDX:friction_upper_idx],
                                                   desired_lateral_jerk)
      if self.use_steering_angle or self.lookahead_lateral_jerk == 0.0:
        self.lookahead_lateral_jerk = 0.0
        self.actual_lateral_jerk = 0.0
        self.lat_accel_friction_factor = 1.0
      self.lateral_jerk_setpoint = self.lat_jerk_friction_factor * self.lookahead_lateral_jerk
      self.lateral_jerk_measurement = self.lat_jerk_friction_factor * self.actual_lateral_jerk

  def update_neural_network(self, CS, params, pid_log, setpoint, measurement, calibrated_pose,
                            desired_lateral_accel, lateral_accel_deadzone):

    if not self.use_nn or self.model_valid:
      return 0.0, pid_log

    # update past data
    roll = params.roll
    if len(calibrated_pose.orientation) > 1:
      pitch = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, pitch)
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(desired_lateral_accel)

    # prepare past and future values
    # adjust future times to account for longitudinal acceleration
    adjusted_future_times = [t + 0.5 * CS.aEgo * (t / max(CS.vEgo, 1.0)) for t in self.nn_future_times]
    past_rolls = [self.roll_deque[min(len(self.roll_deque) - 1, i)] for i in self.history_frame_offsets]
    future_rolls = [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + roll,
                                      np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + pitch) for t in
                    adjusted_future_times]
    past_lateral_accels_desired = [self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque) - 1, i)]
                                   for i in self.history_frame_offsets]
    future_planned_lateral_accels = [np.interp(t, ModelConstants.T_IDXS[:CONTROL_N], self.model_v2.acceleration.y) for t in
                                     adjusted_future_times]

    # compute NNFF error response
    nnff_setpoint_input = [CS.vEgo, setpoint, self.lateral_jerk_setpoint, roll] \
                          + [setpoint] * self.past_future_len \
                          + past_rolls + future_rolls
    # past lateral accel error shouldn't count, so use past desired like the setpoint input
    nnff_measurement_input = [CS.vEgo, measurement, self.lateral_jerk_measurement, roll] \
                             + [measurement] * self.past_future_len \
                             + past_rolls + future_rolls
    torque_from_setpoint = self.torque_from_nn(nnff_setpoint_input)
    torque_from_measurement = self.torque_from_nn(nnff_measurement_input)
    pid_log.error = torque_from_setpoint - torque_from_measurement

    # compute feedforward (same as nn setpoint output)
    error = setpoint - measurement
    friction_input = self.lat_accel_friction_factor * error + self.lat_jerk_friction_factor * self.lookahead_lateral_jerk
    nn_input = [CS.vEgo, desired_lateral_accel, friction_input, roll] \
               + past_lateral_accels_desired + future_planned_lateral_accels \
               + past_rolls + future_rolls
    ff = self.torque_from_nn(nn_input)

    # apply friction override for cars with low NN friction response
    if self.nn_friction_override:
      pid_log.error += self.torque_from_lateral_accel(LatControlInputs(0.0, 0.0, CS.vEgo, CS.aEgo), self.torque_params,
                                                      friction_input,
                                                      lateral_accel_deadzone, friction_compensation=True,
                                                      gravity_adjusted=False)

    return ff, pid_log

  def update_stock_lateral_jerk(self, error):
    accel_error = self.lat_accel_friction_factor * error
    jerk_error = self.lat_jerk_friction_factor * self.actual_lateral_jerk

    return accel_error + jerk_error
