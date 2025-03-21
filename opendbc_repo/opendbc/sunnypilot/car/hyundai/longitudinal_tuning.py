import numpy as np

from opendbc.car import DT_CTRL, structs
from opendbc.car.hyundai.values import HyundaiFlags, CarControllerParams
from opendbc.sunnypilot.car.hyundai.longitudinal_config import Cartuning
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from opendbc.sunnypilot.interpolation_utils import catmull_rom_interp

LongCtrlState = structs.CarControl.Actuators.LongControlState

class JerkOutput:
  def __init__(self, jerk_upper_limit, jerk_lower_limit, cb_upper, cb_lower):
    self.jerk_upper_limit = jerk_upper_limit
    self.jerk_lower_limit = jerk_lower_limit
    self.cb_upper = cb_upper
    self.cb_lower = cb_lower

class HKGLongitudinalTuning:
  """Longitudinal tuning methodology for Hyundai vehicles."""
  def __init__(self, CP: structs.CarParams) -> None:
    self.CP = CP
    self._setup_controllers()
    self._init_state()
    self._setup_car_config()

  def _setup_controllers(self) -> None:
    self.DT_CTRL = DT_CTRL

  def _init_state(self) -> None:
    self.accel_last = 0.0
    self.accel_last_jerk = 0.0
    self.jerk = 0.0
    self.jerk_count = 0.0
    self.jerk_upper_limit = 0.0
    self.jerk_lower_limit = 0.0
    self.cb_upper = self.cb_lower = 0.0
    self.last_decel_time = 0.0
    self.min_cancel_delay = 0.1


  def _setup_car_config(self) -> None:
    self.car_config = Cartuning.get_car_config(self.CP)

  def make_jerk(self, CS: structs.CarState, actuators: structs.CarControl.Actuators) -> float:
    self.jerk_count += 1
    # Handle cancel state to prevent cruise fault
    if not CS.out.cruiseState.enabled or CS.out.gasPressed or CS.out.brakePressed:
      self.accel_last_jerk = 0.0
      self.jerk = 0.0
      self.jerk_count = 0.0
      self.jerk_upper_limit = 0.0
      self.jerk_lower_limit = 0.0
      self.cb_upper = self.cb_lower = 0.0
      return 0.0

    current_accel = actuators.accel
    self.jerk = (current_accel - self.accel_last_jerk)
    self.accel_last_jerk = current_accel

    # the jerk division by ΔT (delta time) leads to too high of values of jerk when ΔT is small, which is not realistic
    # when calculating jerk as a time-based derivative this is a more accurate representation of jerk within OP.

    base_jerk = self.jerk
    xp = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    fp = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
    self.jerk = catmull_rom_interp(base_jerk, xp, fp)
    jerk_max = self.car_config.jerk_limits[1]


    if self.CP.flags & HyundaiFlags.CANFD.value:
      self.jerk_upper_limit = min(max(self.car_config.jerk_limits[0], self.jerk * 2.0), jerk_max)
      self.jerk_lower_limit = min(max(self.car_config.jerk_limits[0], -self.jerk * 4.0), jerk_max)
      self.cb_upper = self.cb_lower = 0.0
    else:
      self.jerk_upper_limit = min(max(self.car_config.jerk_limits[0], self.jerk * 2.0), jerk_max)
      self.jerk_lower_limit = min(max(self.car_config.jerk_limits[0], -self.jerk * 2.0), jerk_max)
      if self.CP.radarUnavailable:
        self.cb_upper = self.cb_lower = 0.0
      else:
        self.cb_upper = self.cb_lower = 0.0
        #if(not Params().get_bool("HyundaiSmootherBraking")) and (CS.out.vEgo > 5.0):
        #  self.cb_upper = float(np.clip(0.20 + actuators.accel * 0.20, 0.0, 1.0))
        #  self.cb_lower = float(np.clip(0.10 + actuators.accel * 0.20, 0.0, 1.0))
        #else:
        #  # When at low speeds, or using smoother braking button, we don't want ComfortBands to be effecting stopping control.
        #  self.cb_upper = self.cb_lower = 0.0

    return self.jerk

  def handle_cruise_cancel(self, CS: structs.CarState):
    """Handle cruise control cancel to prevent faults."""
    if not CS.out.cruiseState.enabled or CS.out.gasPressed or CS.out.brakePressed:
      self.accel_last = 0.0
      return True
    return False

  def calculate_limited_accel(self, actuators: structs.CarControl.Actuators, CS: structs.CarState) -> float:
    """Adaptive acceleration limiting."""
    if self.handle_cruise_cancel(CS):
      return actuators.accel
    self.make_jerk(CS, actuators)
    target_accel = actuators.accel

    if (CS.out.vEgo > 10.0 and target_accel < 0.01):
      brake_ratio = np.clip(abs(target_accel / self.car_config.accel_limits[0]), 0.0, 1.0)
      # Gentler for light braking, more responsive for harder braking
      accel_rate_down = self.DT_CTRL * catmull_rom_interp(brake_ratio,
                                                          np.array([0.25, 0.5, 0.75, 1.0]),
                                                          np.array(self.car_config.brake_response))
      accel = max(target_accel, self.accel_last - accel_rate_down)
    else:
      accel = actuators.accel

    self.accel_last = accel
    return accel

  def calculate_accel(self, actuators: structs.CarControl.Actuators, CS: structs.CarState) -> float:
    """Calculate acceleration with cruise control status handling."""
    if self.handle_cruise_cancel(CS):
      return 0.0
    accel = self.calculate_limited_accel(actuators, CS)
    return float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))


  def apply_tune(self, CP: structs.CarParams) -> None:
    config = self.car_config
    CP.vEgoStopping = config.vego_stopping
    CP.vEgoStarting = config.vego_starting
    CP.stoppingDecelRate = config.stopping_decel_rate
    CP.startAccel = config.start_accel
    CP.startingState = True
    CP.longitudinalActuatorDelay = 0.5


class HKGLongitudinalController:
  """Longitudinal controller which gets injected into CarControllerParams."""

  def __init__(self, CP: structs.CarParams):
    self.CP = CP
    if CP.flags & HyundaiFlagsSP.HKGLONGTUNING:
      self.tuning = HKGLongitudinalTuning(CP)
    else:
      self.tuning = None
    self.jerk = None
    self.jerk_upper = 0.0
    self.jerk_lower = 0.0
    self.cb_upper = 0.0
    self.cb_lower = 0.0
    self.accel = 0.0

  def apply_tune(self, CP: structs.CarParams):
    if CP.flags & HyundaiFlagsSP.HKGLONGTUNING:
      self.tuning.apply_tune(CP)
    else:
      CP.vEgoStopping = 0.5
      CP.vEgoStarting = 0.1
      CP.startingState = True
      CP.startAccel = 1.0
      CP.longitudinalActuatorDelay = 0.5

  def get_jerk(self) -> JerkOutput:
    if self.tuning is not None:
      return JerkOutput(
        self.tuning.jerk_upper_limit,
        self.tuning.jerk_lower_limit,
        self.tuning.cb_upper,
        self.tuning.cb_lower,
      )
    else:
      return JerkOutput(
        self.jerk_upper_limit,
        self.jerk_lower_limit,
        self.cb_upper,
        self.cb_lower,
      )

  def calculate_and_get_jerk(self, actuators: structs.CarControl.Actuators, CS: structs.CarState, long_control_state: LongCtrlState) -> JerkOutput:
    """Calculate jerk based on tuning and return JerkOutput."""
    if self.tuning is not None:
      self.tuning.make_jerk(CS, actuators)
    else:
      jerk_limit = 3.0 if long_control_state == LongCtrlState.pid else 1.0

      self.jerk_upper_limit = jerk_limit
      self.jerk_lower_limit = jerk_limit
      self.cb_upper = 0.0
      self.cb_lower = 0.0
    return self.get_jerk()

  def calculate_accel(self, actuators: structs.CarControl.Actuators, CS: structs.CarState, CP: structs.CarParams) -> float:
    """Calculate acceleration based on tuning and return the value."""
    if CP.flags & HyundaiFlagsSP.HKGLONGTUNING_BRAKING and self.tuning is not None:
      accel = self.tuning.calculate_accel(actuators, CS)
    else:
      accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
    return accel

  def update(self, long_control_state: LongCtrlState, actuators: structs.CarControl.Actuators, CS: structs.CarState,
             CP: structs.CarParams) -> None:
    self.accel = self.calculate_accel(actuators, CS, CP)
    jerk = self.calculate_and_get_jerk(actuators, CS, long_control_state)

    self.jerk_upper = jerk.jerk_upper_limit
    self.jerk_lower = jerk.jerk_lower_limit
    self.cb_upper = jerk.cb_upper
    self.cb_lower = jerk.cb_lower
