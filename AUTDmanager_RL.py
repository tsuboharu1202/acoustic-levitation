
import ctypes
import os
from typing import NoReturn

import numpy as np
from pyautd3 import AUTD3, Controller,SilencerConfig
from pyautd3.gain import Focus
from pyautd3.link.soem import SOEM
from pyautd3.modulation import Sine,Static


class AUTDManager:

    def Init(self) -> None:
        #全部mm
        self.width = 192.0
        self.height = 151.4
        self.workPlaneWidth = 500.0
        self.workPlaneHeight = 500.0
        self.markerSize =70.0
        self.workPlaneDist = 168.0
        self.heightOffset = 2.5
        self.autd = (
        Controller.builder()
        .add_device(AUTD3.from_euler_zyz([-self.width, -self.height, 0.0],[0,0,0]))
        .add_device(AUTD3.from_euler_zyz([0.0, -self.height, 0.0],[0,0,0]))
        .add_device(AUTD3.from_euler_zyz([0.0, 0.0, 0.0],[0,0,0]))
        .add_device(AUTD3.from_euler_zyz([-self.width, 0.0, 0.0],[0,0,0]))
        .add_device(AUTD3.from_euler_zyz([-395.0, 0.0, 200.0],[0,2.0 * 3.1415 * 22.5/360.0,0]))
        .add_device(AUTD3.from_euler_zyz([280.0, 0.0, 30.0],[0,-2.0 * 3.1415 * 22.5/360.0,0]))
        .open_with(
            SOEM()
        )
    )

        self.SetFocus(0,0)
    def SetFocus(self,x,y,amp = 1.0):
        g = Focus(np.array([10.0 + x, -self.height + self.heightOffset, self.workPlaneDist + y])).with_amp(amp)
        m = Static()
        self.autd.send((m, g))
    def SetFocusWithUV(self,u,v,amp = 1.0):
        x = (u - 0.5) * (self.workPlaneWidth - self.markerSize * 2.0)
        y = (v) * (self.workPlaneHeight - self.markerSize * 2.0) + self.markerSize
        self.SetFocus(x,y,amp)