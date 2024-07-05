import datetime
import math
import threading
import time
import typing

import keyboard

import SpoutGL
from OpenGL import GL
from itertools import repeat
import numpy as np
import array
import cv2
from cv2 import aruco
import GenerateArucoImage
from ximea import xiapi

import argparse
import csv
from pythonosc import osc_message_builder
from pythonosc import udp_client

from AUTDManager import AUTDManager

from PlotStream import PlotStream
import cv2


class BallPopController:
    fromName:str
    toName:str
    size:int
    timeCoefficient:float
    cam:xiapi.Camera
    client:udp_client.UDPClient
    renderBuffer: cv2.Mat
    autd:AUTDManager
    prevU:float
    prevV:float
    rawU:float
    rawV:float
    intU:float
    intV:float
    tgtU:float
    tgtV:float
    tgtFromU:float
    tgtFromV:float
    tgtToU:float
    tgtToV:float
    lerpProgress:float
    prevAmp:float
    ampUDiffs:list[float]
    ampVDiffs:list[float]
    times:list[float]
    tgtUs:list[float]
    tgtVs:list[float]
    timePerPosition:float
    hasStart:bool
    hasClosed:bool
    elapsedTime:float
    tgtUId:int
    tgtVId:int
    prevKeyState:bool
    aruco_dict:aruco.Dictionary
    buffer:cv2.Mat
    t:float
    prevTime:float
    f:typing.TextIO
    writer:csv.writer
    missingCount:int



    def __init__(self):
        self.fromName = "input"
        self.toName = "output"
        self.size = 500
        self.timeCoefficient = 0.010
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_exposure(10000)
        self.cam.start_acquisition()
        self.img = xiapi.Image()
        self.ps = PlotStream(100)
        self.client = udp_client.UDPClient("127.0.0.1", 8002)
        self.renderBuffer = None

        self.autd = AUTDManager()
        self.autd.Init()
        self.prevU, self.prevV = 0, 0
        self.rawU,self.rawV = 0,0
        self.intV = 0.0
        self.intU = 0.0
        self.prevAmp = 1.0
        self.prevAmpU = 0.0
        self.prevAmpV = 0.0
        self.tgtU = 0.5
        self.tgtV = 0.5
        self.tgtFromU = 0.5
        self.tgtFromV = 0.5
        self.tgtToU = 0.5
        self.tgtToV = 0.5
        self.lerpProgress = 1.0
        self.prevMatrix = None
        self.missingCount = 0

        self.ampUDiffs = []
        self.ampVDiffs = []

        self.times = []
        self.tgtUs = [0.5, 0.4, 0.3, 0.5, 0.7, 0.6]
        self.tgtVs = [0.5, 0.3, 0.5, 0.7, 0.5, 0.3]
        # tgtUs = [0.5,0.5]
        # tgtVs = [0.05,0.5]
        self.timePerPosition = 5.0
        self.hasStart = False
        self.hasClosed = False
        self.elapsedTime = 0
        self.tgtUId = 0
        self.tgtVId = 0
        self.prevKeyState = False
        self.dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        self.buffer = None
        self.t = 0
        self.prevTime = time.time()
        now = datetime.datetime.now()
        nowStr = now.strftime("%Y%m%d%H%M%S")
        self.f = open('CSV/{}.csv'.format(nowStr), 'w', newline="")
        self.f.__enter__()
        self.writer = csv.writer(self.f)

    def Conv(self, Kd, Kp, t, u):
        i = len(t) - 1
        yval = 0
        for j in range(500):
            index = i - j
            if (index < 1):
                continue
            # print(index)
            yval += 1.0 / Kd * u[index] * np.exp(-Kp / Kd * (t[i] - t[index])) * (t[index] - t[index - 1])
        return yval
    

    def GetOffsetAndAmp(self, u, v, velU, velV, intU, intV, deltaTime, time):

        KdV = 200.0 * self.timeCoefficient
        KpV = 5.0
        KiV = 0.03 / self.timeCoefficient

        KdU = 100.0 * self.timeCoefficient
        KpU = 5.0
        KiU = 0.03 / self.timeCoefficient

        ofsU = self.Conv(KdU, KpU, self.times, self.ampUDiffs)
        intU += ((self.tgtU - u) + ofsU) * deltaTime
        ampU = -velU * KdU  # D
        ampU += -(u - self.tgtU) * KpU  # P
        ampU += intU * KiU  # I
        ampUDiff = min(1.0, max(ampU, -1.0)) - ampU
        self.ampUDiffs.append(ampUDiff)

        ofsV = self.Conv(KdV, KpV, self.times, self.ampVDiffs)
        intV += ((self.tgtV - v) + ofsV) * deltaTime
        ampV = -velV * KdV  # D
        ampV += -(v - self.tgtV) * KpV  # P
        ampV += intV * KiV  # I
        ampVDiff = min(1.0, max(ampV, 0.0)) - ampV
        self.ampVDiffs.append(ampVDiff)

        self.prevAmpU = ampU
        self.prevAmpV = ampV

        if self.tgtV + 0.1 < v and False:
            amp = ampV
        else:
            amp = math.sqrt(ampU ** 2 + max(ampV,0) ** 2)
        dirU = ampU
        dirV = ampV / amp

        deltaU = -0.05 * dirU
        deltaV = 0.0

        self.times.append(time)
        return (deltaU, deltaV, amp, intU, intV)

    def Update(self):

        # process IO
        currentKeyState = keyboard.is_pressed("o")
        if currentKeyState and not self.prevKeyState:
            hasStart = True
        self.prevKeyState = currentKeyState

        self.tgtUId = int(math.floor(self.elapsedTime / self.timePerPosition)) % len(self.tgtUs)
        self.tgtVId = self.tgtUId

        currentTime = time.time()
        deltaTime = currentTime - self.prevTime
        self.prevTime = currentTime

        self.lerpProgress = self.lerpProgress + deltaTime * 0.25
        self.lerpProgress = min(1.0,self.lerpProgress)
        self.tgtU = self.tgtFromU + (self.tgtToU - self.tgtFromU) * self.lerpProgress
        self.tgtV = self.tgtFromV + (self.tgtToV - self.tgtFromV) * self.lerpProgress

        self.cam.get_image(self.img)
        array = self.img.get_image_data_numpy()
        camFetchTime = time.time() - currentTime
        #print("Camera FetchTime:" + str(camFetchTime))
        #print("delta Time:" + str(deltaTime))
        array = np.stack([array for _ in range(3)], axis=0)
        array = np.transpose(array, (1, 2, 0))
        self.buffer = array
        width = self.img.width
        height = self.img.height
        # 位置判定実行
        if type(self.buffer) != type(None):
            # texBuffer = buffer
            texBuffer = cv2.cvtColor(self.buffer, cv2.COLOR_RGBA2RGB)
            texBuffer = cv2.resize(texBuffer, None, fx=1.0, fy=1.2)
            (matrix, isDetected) = GenerateArucoImage.GetMatrixWithMarker(texBuffer, self.dict_aruco, self.size)
            if not isDetected:
                matrix = self.prevMatrix
            else:
                self.prevMatrix = matrix
            if matrix is not None:
                transformed = GenerateArucoImage.ApplyMatrixToImage(texBuffer, matrix, self.size)
                self.renderBuffer = transformed
                # cv2.imwrite("lastPic.png",transformed)
                detected = GenerateArucoImage.DetectBall(transformed, 500)
                if detected is not None:
                    self.missingCount = 0
                    (u, v) = (detected["cx"], detected["cy"])
                    (w, h) = (detected["w"], detected["h"])
                    u = 1.0 - u
                    u = u + 0.0
                    # u = (u - 0.5) * 1.1 + 0.5
                    v = 1.0 - v
                    v
                    self.rawU = u
                    self.rawV = v
                    velU = (u - self.prevU) / deltaTime
                    velV = (v - self.prevV) / deltaTime
                    targetU = u + velU * 3 * self.timeCoefficient
                    targetV = v + velV * 3 * self.timeCoefficient
                    (deltaU, deltaV, amp, self.intU, self.intV) = self.GetOffsetAndAmp(targetU, targetV, velU, velV,
                                                                                       self.intU, self.intV, deltaTime,
                                                                                       currentTime)
                    if self.hasStart:
                        self.elapsedTime += deltaTime
                        self.writer.writerow([self.elapsedTime, u, v, self.tgtU,self.tgtV])
                    # amp = amp * 0.5 + prevAmp * 0.5
                    targetU += deltaU
                    targetV += deltaV

                    self.t += 0.1
                    self.autd.SetFocusWithUV(targetU, targetV, amp)
                    self.ps.addValue(u, v)
                    # ps.draw()
                    self.prevU = u
                    self.prevV = v
                    #print(u, v)
                    #print(amp)
                    self.prevAmp = amp

                    msg = osc_message_builder.OscMessageBuilder(address="/position")
                    msg.add_arg(u)
                    msg.add_arg(v)
                    msg.add_arg(amp)
                    msg = msg.build()
                    self.client.send(msg)
                else:
                    self.missingCount += 1
                    if self.missingCount > 10:
                        self.prevAmp = 0
                        self.intU = 0
                        self.intV = 0
                        self.times = []
                        self.ampVDiffs = []
                        self.ampUDiffs = []
                    

                #print("{},{}".format(transformed.shape, transformed.dtype))

    def RunMainLoopInternal(self):
        while not self.hasClosed:
            self.Update()

    def Close(self):
        print("=========CLOSE==========")
        self.f.__exit__()
        self.hasClosed = True

        self.cam.stop_acquisition()

        # stop communication
        self.cam.close_device()

    def RunMainLoop(self):
        self.thread = threading.Thread(target=self.RunMainLoopInternal)
        self.thread.start()