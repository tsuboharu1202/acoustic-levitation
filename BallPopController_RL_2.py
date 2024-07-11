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

from collections import deque


scale = 40
power_max = 18

# 状態の離散化
velocity_bins = np.array([-np.inf,-5 -3,-2,-1,-0.5, 0,0.5,1,2 , 3,5, np.inf])/scale
distance_bins = np.array([0,0.5,  1,2, 3, 5, 7, 10,15, np.inf])/scale
theta_bins = np.linspace(-np.pi, np.pi, 12)


vx_bins = np.array([-np.inf,-3,-1,-0.5, 0,0.5,1, 3, np.inf])
x_bins = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
y_bins = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# 行動の離散化
forces = np.array([6,7,8,9,10,11, 12, 14])/power_max
angles = np.array([-np.pi/3,-np.pi/4,-np.pi/6,-np.pi/12,0,np.pi/12,np.pi/6 ,np.pi/4,np.pi/3])

radius = 0.05
epsilon = 0.01






class BallPopController_RL:
    angles:list[float]
    forces:list[float]
    gamma:float
    alpha:float
    Q_table:np.ndarray
    state:list
    action:list
    prevState:list
    prevAction:list
    continuous_updates:bool
    last_update_time:float
    total_reward:float

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
        self.total_reward = 0
        self.Q_table = np.load('q_table_pass')
        self.angles = angles
        self.forces = forces


        # 入力と出力の名前設定
        self.fromName = "input"
        self.toName = "output"
        # シミュレーションの画面サイズ設定
        self.size = 500
        # 時間係数（速度調整などに使う）
        self.timeCoefficient = 0.010

        
        # カメラの初期化と設定
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_exposure(10000)
        self.cam.start_acquisition()
        
        # 画像処理用の変数
        self.img = xiapi.Image()
        self.ps = PlotStream(100)
        # OSCプロトコルによるUDP通信クライアント
        self.client = udp_client.UDPClient("127.0.0.1", 8002)
        # 描画用バッファ
        self.renderBuffer = None
        
        # AUTDデバイス管理クラス
        self.autd = AUTDManager()
        self.autd.Init()
        
        # 各種状態の初期化
        self.prevU, self.prevV = 0, 0
        self.rawU, self.rawV = 0, 0
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
        
        # ターゲット座標リスト
        self.ampUDiffs = []
        self.ampVDiffs = []
        self.times = []
        self.tgtUs = [0.5, 0.4, 0.3, 0.5, 0.7, 0.6]
        self.tgtVs = [0.5, 0.3, 0.5, 0.7, 0.5, 0.3]
        self.timePerPosition = 5.0
        self.hasStart = False
        self.hasClosed = False
        self.elapsedTime = 0
        self.tgtUId = 0
        self.tgtVId = 0
        self.prevKeyState = False
        self.dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        
        # データ記録用のファイル設定
        self.buffer = None
        self.t = 0
        self.prevTime = time.time()
        now = datetime.datetime.now()
        nowStr = now.strftime("%Y%m%d%H%M%S")
        self.f = open('CSV/{}.csv'.format(nowStr), 'w', newline="")
        self.f.__enter__()
        self.writer = csv.writer(self.f)

    #Q_tableの更新
    def update_q_table(self,state, action, reward, next_state):
        best_next_action = np.argmax(self.Q_table[next_state])
        best_next_action = (best_next_action // len(self.angles), best_next_action % len(self.angles))
        td_target = reward + self.gamma * self.Q_table[next_state + best_next_action]
        td_error = td_target - self.Q_table[state + action]
        self.Q_table[state + action] += self.alpha * td_error

    def calculate_reward(self,u,v,vx):
        if (u<0.35) & (u>0.65):
            loss = -1000000 

        if v > 0.8:
            return 10000 + loss
        if v > 0.6:
            return 8000 + loss
        if v > 0.4:
            return 4000 + loss
        if v > 0.2:
            return 2000 + loss
        if v < 0.05:
            return -100000 + loss


    def Conv(self, Kd, Kp, t, u):
        i = len(t) - 1
        yval = 0
        for j in range(500):
            index = i - j
            if index < 1:
                continue
            yval += 1.0 / Kd * u[index] * np.exp(-Kp / Kd * (t[i] - t[index])) * (t[index] - t[index - 1])
        return yval



    def choose_action(self,state):
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.forces)), np.random.randint(len(self.angles))
        else:
            idx = np.argmax(self.Q_table[state])
            return idx // len(self.angles), idx % len(self.angles)
        

    def discretize_state(self,x,vx, y):
        vx_bin = np.digitize(vx, velocity_bins) - 1
        y_bin = np.digitize(y, velocity_bins) - 1
        x_bin = np.digitize(x, velocity_bins) - 1
        return vx_bin,x_bin, y_bin




    def GetOffsetAndAmp_RL(self,v,velU,time):
        self.state = self.discretize_state(v,velU)
        action = self.choose_action(self.state)
        force = self.forces[action[0]]
        angle = self.angles[action[1]]
        amp = force


        self.times.append(time)

        return (angle, amp)


        


    def save_q_table(self):
        current_time = time.strftime("%Y%m&d%H%M%S", time.localtime())
        np.save(f'Qtable_{current_time}.npy', self.Q_table)
        print(f'Saved Qtable_{current_time}.npy')


    # システムの更新とアップデート
    def Update_RL(self):
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
        self.lerpProgress = min(1.0, self.lerpProgress)
        self.tgtU = self.tgtFromU + (self.tgtToU - self.tgtFromU) * self.lerpProgress
        self.tgtV = self.tgtFromV + (self.tgtToV - self.tgtFromV) * self.lerpProgress

        self.cam.get_image(self.img)
        array = self.img.get_image_data_numpy()
        camFetchTime = time.time() - currentTime
        array = np.stack([array for _ in range(3)], axis=0)
        array = np.transpose(array, (1, 2, 0))
        self.buffer = array
        width = self.img.width
        height = self.img.height
        if type(self.buffer) != type(None):
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
                detected = GenerateArucoImage.DetectBall(transformed, 500)
                if detected is not None:
                    self.missingCount = 0
                    (u, v) = (detected["cx"], detected["cy"])
                    (w, h) = (detected["w"], detected["h"])
                    u = 1.0 - u
                    v = 1.0 - v
                    self.rawU = u
                    self.rawV = v
                    
                    velU = (u - self.prevU) / deltaTime
                    velV = (v - self.prevV) / deltaTime


                    self.state = self.discretize_state(u,v, velU)
                    self.action = self.choose_action(self.state)
                    amp = self.forces[self.action[0]]
                    angle = self.angles[self.action[1]]


                    reward = self.calculate_reward(u, v, velU)
                    # total_reward += reward
                    # state_action_reward_queue.append((self.prevState, self.prevAction, reward, self.state))


                    # if len(state_action_reward_queue) > delay_steps:
                    #     delayed_state, delayed_action, delayed_reward, delayed_next_state = state_action_reward_queue.popleft()
                    self.update_q_table(self.prevState, self.prevAction, reward, self.state)

                    # targetU = u + velU * 3 * self.timeCoefficient
                    # targetV = v + velself.V * 3 * self.timeCoefficient
                    # (deltaU, deltaV, amp, self.intU, self.intV) = self.GetOffsetAndAmp(targetU, targetV, velU, velV, self.intU, self.intV, deltaTime, currentTime)

                    # (angle, amp) = self.GetOffsetAndAmp_RL(velV,velU,v,u)

                    if self.hasStart:
                        self.elapsedTime += deltaTime
                        self.writer.writerow([self.elapsedTime, u, v, self.tgtU, self.tgtV])
                    # targetU += deltaU
                    # targetV += deltaV

                    self.t += 0.1

                    targetU = u + radius*(-np.sin(angle))
                    targetV = v + radius*(-np.cos(angle))

                    self.autd.SetFocusWithUV(targetU, targetV, amp)
                    self.ps.addValue(u, v)

                    self.prevU = u
                    self.prevV = v
                    self.prevState = self.state
                    self.prevAction = self.action


                    msg = osc_message_builder.OscMessageBuilder(address="/position")
                    msg.add_arg(u)
                    msg.add_arg(v)
                    msg.add_arg(amp)
                    msg = msg.build()
                    self.client.send(msg)

                    self.continuous_updates = True
                    self.last_update_time = time.time()

                else:
                    self.missingCount += 1
                    if self.missingCount > 3 & self.continuous_updates:
                        self.continuous_updates = False
                        self.save_q_table()

                    if self.missingCount > 10:
                        self.prevAmp = 0
                        self.intU = 0
                        self.intV = 0
                        self.times = []
                        self.ampVDiffs = []
                        self.ampUDiffs = []



    # メインループを実行する内部関数
    def RunMainLoopInternal(self):
        while not self.hasClosed:
            self.Update_RL()

    # システムを閉じる処理
    def Close(self):
        print("=========CLOSE==========")
        self.f.__exit__()
        self.hasClosed = True

        self.cam.stop_acquisition()
        self.cam.close_device()

    # メインループを開始する
    def RunMainLoop(self):
        self.thread = threading.Thread(target=self.RunMainLoopInternal)
        self.thread.start()
