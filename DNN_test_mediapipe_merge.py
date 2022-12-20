import tensorflow as tf
import cv2, math
import mediapipe as mp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model

model = load_model("my_model.h5")

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

cap = cv2.VideoCapture(0)
# 啟用姿勢偵測
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

if not cap.isOpened():
    exit()

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    size = frame.shape  # 取得照片影像尺寸
    w = size[1]  # 取得畫面寬度
    h = size[0]  # 取得畫面高度
    results = pose.process(frame)  # 取得姿勢偵測結果
    if results.pose_landmarks:
        x1 = results.pose_landmarks.landmark[11].x * w  # 取得手部末端 x 座標
        x2 = results.pose_landmarks.landmark[12].x * w
        x3 = results.pose_landmarks.landmark[13].x * w
        x4 = results.pose_landmarks.landmark[14].x * w
        x5 = results.pose_landmarks.landmark[15].x * w
        x6 = results.pose_landmarks.landmark[16].x * w
        x7 = results.pose_landmarks.landmark[23].x * w
        x8 = results.pose_landmarks.landmark[24].x * w
        x9 = results.pose_landmarks.landmark[25].x * w
        x10 = results.pose_landmarks.landmark[26].x * w
        x11 = results.pose_landmarks.landmark[27].x * w
        x12 = results.pose_landmarks.landmark[28].x * w
        y1 = results.pose_landmarks.landmark[11].y * h  # 取得手部末端 y 座標
        y2 = results.pose_landmarks.landmark[12].y * h
        y3 = results.pose_landmarks.landmark[13].y * h
        y4 = results.pose_landmarks.landmark[14].y * h
        y5 = results.pose_landmarks.landmark[15].y * h
        y6 = results.pose_landmarks.landmark[16].y * h
        y7 = results.pose_landmarks.landmark[23].y * h
        y8 = results.pose_landmarks.landmark[24].y * h
        y9 = results.pose_landmarks.landmark[25].y * h
        y10 = results.pose_landmarks.landmark[26].y * h
        y11 = results.pose_landmarks.landmark[27].y * h
        y12 = results.pose_landmarks.landmark[28].y * h

        B_C = [x4, y4, x6, y6]
        B_A = [x4, y4, x2, y2]
        A_B = [x2, y2, x4, y4]
        A_G = [x2, y2, x8, y8]
        G_A = [x8, y8, x2, y2]
        G_I = [x8, y8, x10, y10]
        I_G = [x10, y10, x8, y8]
        I_K = [x10, y10, x12, y12]
        E_F = [x3, y3, x5, y5]
        E_D = [x3, y3, x1, y1]
        D_E = [x1, y1, x3, y3]
        D_H = [x1, y1, x7, y7]
        H_D = [x7, y7, x1, y1]
        H_J = [x7, y7, x9, y9]
        J_H = [x9, y9, x7, y7]
        J_L = [x9, y9, x11, y11]

        ang_1 = angle(B_C, B_A)
        ang_2 = angle(A_B, A_G)
        ang_3 = angle(G_A, G_I)
        ang_4 = angle(I_G, I_K)
        ang_5 = angle(E_F, E_D)
        ang_6 = angle(D_E, D_H)
        ang_7 = angle(H_D, H_J)
        ang_8 = angle(J_H, J_L)

        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    cap.release()
    cv2.destroyAllWindows()

train = np.array([[int(ang_1), ang_2, ang_3,ang_4,ang_5,ang_6,ang_7,ang_8]])

pre = model.predict(train)
print(pre)
