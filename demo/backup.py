import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import sys
import numpy as np
from math import sin, cos

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Average
# from keras.models import Model, Input
from keras.models import Model
from keras.layers import Input

sys.path.append("..")
from lib.FSANET_model import *


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=120):
    print(yaw, roll, pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 6)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 6)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 5)

    return img


def draw_results_ssd(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network,
                     time_plot):
    size = 100
    # loop over the detections
    global output, img
    if detected.shape[2] > 0:
        for i in range(0, detected.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detected[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")

                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY

                x2 = x1 + w
                y2 = y1 + h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)
                cv2.rectangle(input_img, (50,160), (565,390), (0, 0, 0), -1)
                # conf = float(p_result)
                # confy = {round(conf, 2)}
                cv2.putText(input_img, (f'x = {str(round(p_result[0][0], 4))}'), (80, 225), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                cv2.putText(input_img, (f'y = {str(round(p_result[0][1], 4))}'), (80, 295), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(input_img, (f'z = {str(round(p_result[0][2], 4))}'), (80, 365), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                cv2.rectangle(input_img, (50, 130), (720, 5), (0, 0, 0), -1)
                if p_result[0][1] < -15:
                    text = "Looking Right"
                    # cv2.putText(input_img, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(input_img, text, (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                elif p_result[0][0] > 15:
                    text = "Looking Left"
                    cv2.putText(input_img, text, (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                elif p_result[0][1] < -15:
                    text = "Looking Down"
                    cv2.putText(input_img, text, (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                else:
                    text = "Head Position is ok "
                    cv2.putText(input_img, text, (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

                img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])

                input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img

                # ims = cv2.imread('img1.jpg')
                img = input_img[yw1+280:yw2 -320, xw1+280:xw2 -280]
                size = 50
                logo = cv2.resize(img, (size, size))
                rows, cols, channels = logo.shape
                cv2.rectangle(input_img, (700, rows+450), (50, cols-200), (0, 0, 255), 5)

                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                # rows0, cools0 = logo.shape

                roi = input_img[450:rows+450, 50:cols+50]
                roi[np.where(mask)] = 0
                roi += logo

    else:
        cv2.imshow("result", input_img)

    return input_img, img


def main():
    global faces
    try:
        os.mkdir('./img')
    except OSError:
        pass

    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    # detector = MTCNN()

    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = ''  # make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 1  # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
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

    print('Loading models ...')

    weight_file1 = '/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = '/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["face_detector",
                                  "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # capture video
    cap = cv2.VideoCapture('/home/ean/workspace/PERSONAL/HEAD_ESTIMATION_2/demo/head_position_cut.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640 * 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('your_video6.avi', fourcc, 20.0, size)

    print('Start detecting pose ...')
    detected_pre = np.empty((1, 1, 1))

    while(cap.isOpened()):

        # get video frame
        ret, input_img = cap.read()

        img_idx = img_idx + 1
        img_h, img_w, _ = np.shape(input_img)

        if img_idx == 1 or img_idx % skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0

            # detect faces using LBP detector
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            # detected = face_cascade.detectMultiScale(gray_img, 1.1)
            # detected = detector.detect_faces(input_img)
            # pass the blob through the network and obtain the detections and
            # predictions
            blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detected = net.forward()

            if detected_pre.shape[2] > 0 and detected.shape[2] == 0:
                detected = detected_pre

            faces = np.empty((detected.shape[2], img_size, img_size, 3))

            input_img, img = draw_results_ssd(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection,
                                         time_network, time_plot)
            # create kernel for image dilation
            kernel = np.ones((3, 3), np.uint8)


            out.write(input_img)
            cv2.imwrite('img/' + str(img_idx) + '.png', img)


        if detected.shape[2] > detected_pre.shape[2] or img_idx % (skip_frame * 3) == 0:
            detected_pre = detected

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    main()
