import math
import sys
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import  BallPopController_RL
import time

class BallPopRenderer:
    # 初期化メソッド、コントローラーとの連携やGUIコンポーネントのセットアップを行う
    def __init__(self, controller):
        self.frameCount = 0
        self.controller = controller
        self.isRecording = False
        self.recButtonText = "Begin Recording"

        self.root = Tk()  # メインウィンドウの作成
        self.root.title("Hoge")  # ウィンドウタイトルの設定
        self.root.geometry("1000x400")  # ウィンドウサイズの設定

        # 閉じるボタンの作成と配置
        self.closeButton = ttk.Button(self.root, text="Close", command=self.onClose)
        self.closeButton.pack()

        # 録画ボタンの作成と配置
        self.recButton = ttk.Button(self.root, text=self.recButtonText, command=self.toggleRec)
        self.recButton.pack()

        # ウィンドウのクローズイベントを無効化（クローズボタンの動作をカスタマイズ）
        self.root.protocol("WM_DELETE_WINDOW", self.disable_event)

        # キャンバス（画像表示エリア）の作成と配置
        self.canvas = Canvas(self.root, width=100, height=100)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.show_position)  # マウスクリックイベントのバインド

        # 状態表示用のラベルの作成と配置
        self.statsLabel = ttk.Label(self.root, text="Label")
        self.statsLabel.pack()

        self.update()  # 画面更新の開始
        self.root.mainloop()  # GUIイベントループの開始
        self.onClose()  # GUIクローズ時の処理

    # マウスクリックイベントのハンドラ、クリックされた位置に基づいてコントローラの目標位置を更新
    def show_position(self, event):
        x, y = event.x, event.y  # クリックされた位置の取得
        print(x, y)  # デバッグ出力
        # コントローラの目標位置を更新
        self.controller.tgtFromU, self.controller.tgtFromV = self.controller.tgtToU, self.controller.tgtToV
        self.controller.tgtToU, self.controller.tgtToV = (self.controller.size - x) / float(self.controller.size), (self.controller.size - y) / float(self.controller.size)
        self.controller.lerpProgress = 0  # 補間進行度のリセット

    # 録画状態のトグル
    def toggleRec(self):
        if self.isRecording:
            self.isRecording = False
            self.EndRecord()  # 録画終了
            self.recButtonText = "Begin Recording"
        else:
            self.isRecording = True
            self.StartRecord()  # 録画開始
            self.recButtonText = "End Recording"
        self.recButton.configure(text=self.recButtonText)  # ボタンのテキスト更新

    # 録画開始処理
    def StartRecord(self):
        currentTime = time.time()
        path = "{}.mp4".format(str(math.floor(currentTime)))  # ファイル名の生成
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # コーデックの指定
        # ビデオライターの作成（ファイルへの書き出しを開始）
        self.videoWriter = cv2.VideoWriter(path, fourcc, 30, (self.controller.size, self.controller.size))

    # 録画終了処理
    def EndRecord(self):
        if self.videoWriter:
            self.videoWriter.release()  # ビデオライターのリリース

    # 画面更新処理
    def update(self):
        if self.controller.renderBuffer is not None:
            img = self.controller.renderBuffer
            # 画像上に各種マーカーとログを描画
            if self.isRecording and self.videoWriter is not None:
                self.videoWriter.write(img)  # 録画中は画像をファイルに書き込む

            image_pil = Image.fromarray(img)  # OpenCV画像をPIL形式に変換
            self.image_tk = ImageTk.PhotoImage(image_pil, master=self.root)  # PIL画像をImageTk形式に変換
            # キャンバスのサイズ調整と画像の描画
            self.canvas.configure(width=img.shape[1], height=img.shape[0])
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.image_tk, anchor="nw")
        # フレーム数と振幅のログをラベルに表示
        statsStr = "[frame]{}[amp]{}".format(str(self.frameCount), str(self.controller.prevAmp))
        self.statsLabel.configure(text=statsStr)
        self.frameCount += 1
        self.loopId = self.root.after(16, self.update)  # 16ミリ秒後に再度updateメソッドを呼び出す

    # ウィンドウクローズ時の処理
    def onClose(self):
        self.controller.Close()  # コントローラのクローズ処理
        self.root.after_cancel(self.loopId)  # 更新処理のキャンセル
        sys.exit()  # システム終了
