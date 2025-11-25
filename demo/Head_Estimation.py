import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU for TensorFlow

import cv2
import sys
import numpy as np
from math import sin, cos
from collections import deque

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Average

sys.path.append("..")
from lib.FSANET_model import *


# ------------------------------
# Smoothing buffers for angles
# ------------------------------
SMOOTH_WINDOW = 10
yaw_hist   = deque(maxlen=SMOOTH_WINDOW)
pitch_hist = deque(maxlen=SMOOTH_WINDOW)
roll_hist  = deque(maxlen=SMOOTH_WINDOW)


def rounded_rectangle(img, pt1, pt2, color, thickness, radius=15):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw 4 straight lines
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)

    # Draw 4 arcs (rounded corners)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50):
    # yaw, pitch, roll in degrees
    pitch = pitch * np.pi / 180.0
    yaw = -(yaw * np.pi / 180.0)
    roll = roll * np.pi / 180.0

    h, w = img.shape[:2]
    if tdx is None:
        tdx = w / 2
    if tdy is None:
        tdy = h / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis (green)
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (blue)
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 6)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 6)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 5)
    return img


def draw_results_ssd(detected, input_img, faces, ad, img_size, img_w, img_h,
                     model, time_detection, time_network, time_plot):
    global yaw_hist, pitch_hist, roll_hist

    overlay = input_img.copy()

    if detected.shape[2] > 0:
        for i in range(detected.shape[2]):
            confidence = detected[0, 0, i, 2]
            if confidence <= 0.5:
                continue

            (h0, w0) = input_img.shape[:2]
            box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
            (startX, startY, endX, endY) = box.astype("int")

            x1, y1 = startX, startY
            w = endX - startX
            h = endY - startY
            x2 = x1 + w-20
            y2 = y1 + h-20
            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)

            # face crop for network
            face_crop = cv2.resize(input_img[yw1:yw2 + 0, xw1:xw2 + 1, :],
                                   (img_size, img_size))
            face_crop = cv2.normalize(face_crop, None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX)
            faces[i, :, :, :] = face_crop

            face = np.expand_dims(face_crop, axis=0)
            yaw_raw, pitch_raw, roll_raw = model.predict(face)[0]

            # -------- Smoothing --------
            yaw_hist.append(yaw_raw)
            pitch_hist.append(pitch_raw)
            roll_hist.append(roll_raw)

            yaw = float(np.mean(yaw_hist))
            pitch = float(np.mean(pitch_hist))
            roll = float(np.mean(roll_hist))

            # -------- Orientation text --------
            if yaw > 15:
                orientation_text = "Looking Left"
            elif yaw < -15:
                orientation_text = "Looking Right"
            elif pitch > 15:
                orientation_text = "Looking Up"
            elif pitch < -15:
                orientation_text = "Looking Down"
            else:
                orientation_text = "Head Position is ok"

            # ---------- UI panel ----------
            # big black panel
            cv2.rectangle(overlay, (50, 50), (580, 250), (0, 0, 0), -1)
            alpha = 0.6
            input_img = cv2.addWeighted(overlay, alpha, input_img, 1 - alpha, 0)

            # orientation text
            cv2.putText(input_img, orientation_text, (70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

            # yaw/pitch/roll values
            cv2.putText(input_img, f"x = {yaw:.4f}", (70, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
            cv2.putText(input_img, f"y = {pitch:.4f}", (70, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(input_img, f"z = {roll:.4f}", (70, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

            # ---------- pose axis on face ----------
            face_region = input_img[yw1:yw2 + 1, xw1:xw2 + 1, :]
            face_axis = draw_axis(face_region, yaw, pitch, roll)
            input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = face_axis

            # ---------- small preview (top-left) ----------
            preview = cv2.resize(face_axis, (280, 280))
            input_img[260:260 + 280, 50:50 + 280] = preview
            # Apply rounded red frame
            rounded_rectangle(input_img, (50, 260), (330, 540), (102, 102, 255), 2)

    return input_img, input_img


def main():
    global yaw_hist, pitch_hist, roll_hist

    os.makedirs("./img", exist_ok=True)

    img_size = 64
    stage_num = [3, 3, 3]
    lambda_d = 1
    img_idx = 0
    detected = ""
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 1
    ad = 0.1

    # ------------- FSA-Net models -------------
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    num_classes = 3
    image_size = 64
    num_primcaps = 7 * 3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    print("Loading models ...")
    model1.load_weights("/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5")
    print("Finished loading model 1.")
    model2.load_weights("/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5")
    print("Finished loading model 2.")
    model3.load_weights("/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5")
    print("Finished loading model 3.")

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)
    x2 = model2(inputs)
    x3 = model3(inputs)
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # ------------- Face detector (SSD) -------------
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # ------------- Video IO -------------
    cap = cv2.VideoCapture("/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/demo/head_position_cut.mp4")
    if not cap.isOpened():
        print("Error: cannot open video file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_head_pose.mp4", fourcc, fps, size)

    print("Start detecting pose ...")
    detected_pre = np.empty((1, 1, 1))

    while cap.isOpened():
        ret, input_img = cap.read()
        if not ret:
            break

        img_idx += 1
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0

            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()

            if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
                detected = detected_pre

            faces = np.empty((detected.shape[2], img_size, img_size, 3))

            input_img, img_vis = draw_results_ssd(
                detected, input_img, faces, ad, img_size, img_w, img_h, model,
                time_detection, time_network, time_plot
            )

            out.write(input_img)
            cv2.imwrite(f"img/{img_idx:06d}.png", img_vis)

        if detected.shape[2] > detected_pre.shape[2] or img_idx % (skip_frame * 3) == 0:
            detected_pre = detected

    cap.release()
    out.release()
    print("Finished. Saved to output_head_pose.mp4")


if __name__ == "__main__":
    main()