from cereal import car
from common.realtime import DT_CTRL
from common.numpy_fast import interp, clip
from common.conversions import Conversions as CV
from selfdrive.car import apply_std_steer_torque_limits, create_gas_interceptor_command
from selfdrive.car.gm import gmcan
from selfdrive.car.gm.values import DBC, NO_ASCM, CanBus, CarControllerParams
from opendbc.can.packer import CANPacker
from selfdrive.controls.lib.drive_helpers import V_CRUISE_ENABLE_MIN
from selfdrive.ntune import ntune_scc_get
from common.params import Params

VisualAlert = car.CarControl.HUDControl.VisualAlert
GearShifter = car.CarState.GearShifter

CREEP_SPEED = 2.5   # 4km

class CarController():

  def get_lead(self, sm):
    radar = sm['radarState']
    if radar.leadOne.status:
      return radar.leadOne
    return None

  def __init__(self, dbc_name, CP, VM):
    self.apply_steer_last = 0
    self.comma_pedal = 0.0
    self.accel = 0

    self.lka_steering_cmd_counter_last = -1
    self.lka_icon_status_last = (False, False)
    #self.RestartForceAccel = Params().get_bool('RestartForceAccel')

    self.params = CarControllerParams(CP)

    self.packer_pt = CANPacker(DBC[CP.carFingerprint]['pt'])
    #self.packer_obj = CANPacker(DBC[CP.carFingerprint]['radar'])
    #self.packer_ch = CANPacker(DBC[CP.carFingerprint]['chassis'])

  def update(self, c, enabled, CS, frame, controls, actuators,
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert):

    P = self.params

    # Send CAN commands.
    can_sends = []

    # Steering (50Hz)
    # Avoid GM EPS faults when transmitting messages too close together: skip this transmit if we just received the
    # next Panda loopback confirmation in the current CS frame.
    if CS.lka_steering_cmd_counter != self.lka_steering_cmd_counter_last:
      self.lka_steering_cmd_counter_last = CS.lka_steering_cmd_counter
    elif (frame % P.STEER_STEP) == 0:
      lkas_enabled = c.active and not (CS.out.steerFaultTemporary or CS.out.steerFaultPermanent) and CS.out.vEgo > P.MIN_STEER_SPEED
      if lkas_enabled:
        new_steer = int(round(actuators.steer * P.STEER_MAX))
        apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, P)
      else:
        apply_steer = 0

      self.apply_steer_last = apply_steer
      # GM EPS faults on any gap in received message counters. To handle transient OP/Panda safety sync issues at the
      # moment of disengaging, increment the counter based on the last message known to pass Panda safety checks.
      idx = (CS.lka_steering_cmd_counter + 1) % 4

      can_sends.append(gmcan.create_steering_control(self.packer_pt, CanBus.POWERTRAIN, apply_steer, idx, lkas_enabled))

      self.accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)

      if CS.CP.enableGasInterceptor:
        # 이것이 없으면 저속에서 너무 공격적입니다.
        if c.active and CS.adaptive_Cruise and CS.out.vEgo > V_CRUISE_ENABLE_MIN / CV.MS_TO_KPH:

          acc_mult = interp(CS.out.vEgo, [0., 18.0 * CV.KPH_TO_MS, 30 * CV.KPH_TO_MS, 60 * CV.KPH_TO_MS, 80 * CV.KPH_TO_MS],
                                         [0.18, 0.20, 0.23, 0.24, 0.25])
          #pedal_offset = interp(CS.out.vEgo, [0.0, CREEP_SPEED, CREEP_SPEED*2], [-.5, 0.15, 0.2])

          ## =============================================== ##

          pedal_command = acc_mult * actuators.accel

          ## ================================================ ##
          #pedal_command = PEDAL_SCALE * actuators.accelf
          self.comma_pedal = clip(pedal_command, 0., 1.)

          """acc_mult = interp(CS.out.vEgo, [0., 18.0 * CV.KPH_TO_MS, 30 * CV.KPH_TO_MS, 40 * CV.KPH_TO_MS],
                            [0.17, 0.24, 0.265, 0.24])
          start_boost = interp(CS.out.vEgo, [0.0, CREEP_SPEED, 1.5*CREEP_SPEED], [-0.4, 0.20, 0.15])
          self.comma_pedal = clip(acc_mult * (actuators.accel + start_boost), 0., 1.)"""

        elif not c.active or not CS.adaptive_Cruise or CS.out.vEgo <= V_CRUISE_ENABLE_MIN / CV.MS_TO_KPH:
          self.comma_pedal = 0.0

        if (frame % 4) == 0:
          idx = (frame // 4) % 4
          can_sends.append(create_gas_interceptor_command(self.packer_pt, self.comma_pedal, idx))

    # Show green icon when LKA torque is applied, and
    # alarming orange icon when approaching torque limit.
    # If not sent again, LKA icon disappears in about 5 seconds.
    # Conveniently, sending camera message periodically also works as a keepalive.
    lka_active = CS.lkas_status == 1
    lka_critical = lka_active and abs(actuators.steer) > 0.9
    lka_icon_status = (lka_active, lka_critical)
    if frame % P.CAMERA_KEEPALIVE_STEP == 0 or lka_icon_status != self.lka_icon_status_last:
      steer_alert = hud_alert in (VisualAlert.steerRequired, VisualAlert.ldw)
      can_sends.append(gmcan.create_lka_icon_command(CanBus.SW_GMLAN, lka_active, lka_critical, steer_alert))
      self.lka_icon_status_last = lka_icon_status

    new_actuators = actuators.copy()
    new_actuators.steer = self.apply_steer_last / P.STEER_MAX
    new_actuators.accel = self.accel
    new_actuators.gas = self.comma_pedal

    return new_actuators, can_sends
