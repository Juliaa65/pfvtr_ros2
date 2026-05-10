#!/usr/bin/env python3
from geometry_msgs.msg import Twist


class Controller:

    def __init__(self):
        self.alignment = 0
        self.uncertainty = 0
        self.useUncertainty = True
        self.turnGain = 1.0  #turn 0.1 rad per each pixel of error
        self.velocityGain = 1.0 # 1 is same speed as thought map, less is slower more is faster
        # 0.0 disables the override and falls back to msg.linear.x * velocityGain.
        self.constantVelocity = 0.0
        # Linear ramp distance (m): commanded linear.x and angular.z are scaled
        # by clip(distance_remaining / decelDistance, 0, 1) so the robot
        # decelerates smoothly into the goal. <=0 disables the ramp.
        self.decelDistance = 1.0
        # Pushed by ROS subscription to /pfvtr/repeat/distance_remaining.
        # Default +inf so the ramp is a no-op when no repeat is in progress.
        self.distanceRemaining = float("inf")

    def process(self, msg):
        correction = self.alignment * self.turnGain # angle = px * angle/pixel
        # if self.useUncertainty:
        #     correction = correction * (1 - self.uncertainty)
        # Constant-velocity override: when set, the recorded velocity profile
        # is ignored and a fixed forward speed is commanded instead.
        if self.constantVelocity != 0.0:
            base_x = self.constantVelocity
        else:
            base_x = msg.linear.x * self.velocityGain
        # End-of-map deceleration: scale linear.x AND angular.z by the same
        # factor so the robot smoothly stops in place (no rotation at zero
        # forward speed).
        if self.decelDistance > 0.0:
            decel_scale = max(0.1, min(1.0, self.distanceRemaining / self.decelDistance))
        else:
            decel_scale = 1.0
        out = Twist()
        out.linear.x = base_x * decel_scale
        out.linear.y = msg.linear.y * self.velocityGain
        out.linear.z = msg.linear.z * self.velocityGain
        out.angular.x = msg.angular.x * self.turnGain
        out.angular.y = msg.angular.y * self.turnGain
        out.angular.z = ((msg.angular.z * self.velocityGain) + correction) * decel_scale
        return out

    def reconfig(self,cfg):
        self.useUncertainty = bool(cfg.get("use_uncertainty", self.useUncertainty))
        self.turnGain = float(cfg.get("turn_gain", self.turnGain))
        self.velocityGain = float(cfg.get("velocity_gain", self.velocityGain))
        self.constantVelocity = float(cfg.get("constant_velocity", self.constantVelocity))
        self.decelDistance = float(cfg.get("decel_distance", self.decelDistance))

    def set_distance_remaining(self, value: float):
        self.distanceRemaining = float(value)

    def correction(self,msg):
        self.alignment = msg.output #Px
        # self.uncertainty = msg.uncertainty

