#!/usr/bin/env python3
from geometry_msgs.msg import Twist


class Controller:

    def __init__(self):
        self.alignment = 0
        self.uncertainty = 0
        self.useUncertainty = True
        self.turnGain = 0.5  #turn 0.1 rad per each pixel of error
        self.velocityGain = 1 # 1 is same speed as thought map, less is slower more is faster

    def process(self, msg):
        correction = self.alignment * self.turnGain # angle = px * angle/pixel
        # if self.useUncertainty:
        #     correction = correction * (1 - self.uncertainty)
        out = Twist()
        out.linear.x = msg.linear.x * self.velocityGain
        out.linear.y = msg.linear.y * self.velocityGain
        out.linear.z = msg.linear.z * self.velocityGain
        out.angular.x = msg.angular.x * self.turnGain 
        out.angular.y = msg.angular.y * self.turnGain
        out.angular.z = msg.angular.z + correction
        return out

    def reconfig(self,cfg):
        self.useUncertainty = bool(cfg.get("use_uncertainty", self.useUncertainty))
        self.turnGain = float(cfg.get("turn_gain", self.turnGain))
        self.velocityGain = float(cfg.get("velocity_gain", self.velocityGain))

    def correction(self,msg):
        self.alignment = msg.output #Px
        # self.uncertainty = msg.uncertainty

