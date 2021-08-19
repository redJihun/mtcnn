import os

import numpy
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute, PReLU
from tensorflow.keras.models import Model

import numpy as np

import cv2

import time


def Pnet(weight_path='model12old.h5'):
    input = Input(shape=[None, None, 3])
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model


def Rnet(weight_path='model24.h5'):
    input = Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)

    return model


def Onet(weight_path='model48.h5'):
    input = Input(shape = [48,48,3])
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)
    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


'''
Function:
    change rectangles into squares (matrix version)
Input:
    rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    squares: same as input
'''
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T
    return rectangles


'''
Function:
    apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
Input:
    rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
    rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'iom':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


'''
Function:
    Detect face position and calibrate bounding box on 12net feature map(matrix version)
Input:
    cls_prob : softmax feature map for face classify
    roi      : feature map for regression
    out_side : feature map's largest size
    scale    : current input image scale in multi-scales
    width    : image's origin width
    height   : image's origin height
    threshold: 0.6 can have 99% recall rate
'''
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    in_side = 2*out_side+11
    stride = 0
    if out_side != 1:
        stride = float(in_side-12)/(out_side-1)
    (x,y) = np.where(cls_prob>=threshold)
    boundingbox = np.array([x,y]).T
    bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1,bb2),axis = 1)
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T
    boundingbox = boundingbox + offset*12.0*scale
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])

    # print result(중간 성능)
    return NMS(pick,0.3,'iou')


'''
Function:
    Filter face position and calibrate bounding box on 12net's output
Input:
    cls_prob  : softmax feature map for face classify
    roi_prob  : feature map for regression
    rectangles: 12net's predict
    width     : image's origin width
    height    : image's origin height
    threshold : 0.6 can have 97% recall rate
Output:
    rectangles: possible face positions
'''
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])

    # print result(중간 성능)
    return NMS(pick,0.3,'iou')


'''
Function:
    Filter face position and calibrate bounding box on 12net's output
Input:
    cls_prob  : cls_prob[1] is face possibility
    roi       : roi offset
    pts       : 5 landmark
    rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
    width     : image's origin width
    height    : image's origin height
    threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
    rectangles: face positions and landmarks
'''
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    # pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    # pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
    # pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    # pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
    # pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    # pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    # pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
    # pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    # pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
    # pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])

    # print result(중간 성능)
    return NMS(pick,0.3,'iom')


'''
Function:
    calculate multi-scale and limit the maxinum side to 1000 
Input: 
    img: original image
Output:
    pr_scale: limit the maxinum side to 1000, < 1.0
    scales  : Multi-scale
'''
def calculateScales(img):
    caffe_img = img.copy()
    pr_scale = 1.0
    h,w,ch = caffe_img.shape
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0/max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    #multi-scale
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales


def filter_face_48net_newdef(cls_prob,roi,pts,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,1]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,3]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,6]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,8]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    # print (pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])

    # print result(중간 성능)
    return NMS(pick,0.3,'idsom')


def detectFace(img, bbox_label, dataset, threshold):
    # load pretrained model
    pnet = Pnet(r'12net.h5')
    rnet = Rnet(r'24net.h5')
    onet = Onet(r'48net.h5')

    # Make zero-mean & (-1,1) scaled
    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = calculateScales(img)

    out = []

    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        ouput = pnet.predict(input)
        out.append(ouput)

    image_num = len(scales)
    rectangles = []

    # pnet =============================================================================================================
    t1 = time.time()
    for i in range(image_num):
        cls_prob = out[i][0][0][:, :, 1] # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)

        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = detect_face_12net(cls_prob, roi, out_side, 1/scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)

    t2 = time.time()
    time1 = t2 - t1
    pnet_result, _ = calculate_iou(rectangles, bbox_label, dataset)
    print("pnet_result: {}".format(pnet_result))
    # ==================================================================================================================

    # pnms =============================================================================================================
    t1 = time.time()
    rectangles = NMS(rectangles, 0.7, 'iou')
    t2 = time.time()
    time2 = t2 - t1
    pnms_result, _ = calculate_iou(rectangles, bbox_label, dataset)
    print("pnms_result: {}".format(pnms_result))
    # ==================================================================================================================

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    out = []
    predict_24_batch = []
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (24, 24))
        predict_24_batch.append(scale_img)
        crop_number += 1

    predict_24_batch = np.array(predict_24_batch)

    # rnet =============================================================================================================
    t1 = time.time()
    out = rnet.predict(predict_24_batch)

    cls_prob = out[0]
    cls_prob = np.array(cls_prob)
    roi_prob = out[1]
    roi_prob = np.array(roi_prob)
    rectangles = filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
    t2 = time.time()

    time3 = t2 - t1
    rnet_result, _ = calculate_iou(rectangles, bbox_label, dataset)
    print("rnet_result: {}".format(rnet_result))
    # ==================================================================================================================

    # rnet =============================================================================================================
    t1 = time.time()
    rectangles = NMS(rectangles, 0.7, 'iou')
    t2 = time.time()

    time4 = t2 - t1
    rnms_result, _ = calculate_iou(rectangles, bbox_label, dataset)
    print("rnms_result: {}".format(rnms_result))
    # ==================================================================================================================

    if len(rectangles) == 0:
        return rectangles

    crop_number = 0
    predict_batch = []

    for rectangle in rectangles:
        # print('calculating net 48 crop_number:', crop_number)
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img, (48, 48))
        predict_batch.append(scale_img)
        crop_number += 1

    predict_batch = np.array(predict_batch)

    # onet =============================================================================================================
    t1 = time.time()
    output = onet.predict(predict_batch)
    cls_prob = output[0]
    roi_prob = output[1]
    pts_prob = output[2]  # index
    rectangles = filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    # rectangles = filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    t2 = time.time()

    time5 = t2 - t1
    onet_result, num_of_objects = calculate_iou(rectangles, bbox_label, dataset)
    print("onet_result: {}".format(onet_result))
    # ==================================================================================================================

    return pnet_result, pnms_result, rnet_result, rnms_result, onet_result, time1, time2, time3, time4, time5, num_of_objects


def calculate_iou(rects, bbox_label, ds):
    confidences = []
    ious = []
    true_positives = []
    false_positives = []
    rectangles = rects.copy()
    if ds == 'c':
        num_of_object = 1
        try:
            for rectangle in rectangles:

                # bbox가 라벨box의 범위를 아예 벗어나는 경우 iou를 계산하지 않음
                if (min(rectangle[2], int(bbox_label[1]) + int(bbox_label[3])) - max(rectangle[0],
                                                                                     int(bbox_label[1]))) < 0 or \
                        (min(rectangle[3], int(bbox_label[2]) + int(bbox_label[4])) - max(rectangle[1],
                                                                                          int(bbox_label[2]))) < 0:
                    continue

                # 분자 계산 = 라벨과 예측 bbox의 교집합 넓이
                numerator = (min(rectangle[2], int(bbox_label[1]) + int(bbox_label[3])) - max(rectangle[0],
                                                                                              int(bbox_label[1]))) * \
                            (min(rectangle[3], int(bbox_label[2]) + int(bbox_label[4])) - max(rectangle[1],
                                                                                              int(bbox_label[2])))
                # 분모 계산 = 라벨과 예측 bbox의 합집합 넓이(= 라벨bbox넓이 + 예측bbox넓이 - 교집합넓이)
                denominator = (int(bbox_label[3]) * int(bbox_label[4])) + (
                            (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])) - numerator
                # IoU = 교집합넓이 / 합집합넓이
                iou = numerator / denominator
                if iou >= 0.5:
                    confidences.append(rectangle[4])
                    ious.append(iou)
                    true_positives.append(1)
                    false_positives.append(0)
                    rectangles.remove(rectangle)
                    break
        except:
            pass
    else:
        num_of_object = len(bbox_label)
        for lbl in bbox_label:
            try:
                for rectangle in rectangles:
                    # bbox가 라벨box의 범위를 아예 벗어나는 경우 iou를 계산하지 않음
                    if (min(rectangle[2], int(lbl[0]) + int(lbl[2])) - max(rectangle[0], int(lbl[0]))) < 0 or \
                            (min(rectangle[3], int(lbl[1]) + int(lbl[3])) - max(rectangle[1], int(lbl[1]))) < 0:
                        continue

                    else:
                        # 분자 계산 = 라벨과 예측 bbox의 교집합 넓이
                        numerator = (min(rectangle[2], int(lbl[0]) + int(lbl[2])) - max(rectangle[0], int(lbl[0]))) * \
                                    (min(rectangle[3], int(lbl[1]) + int(lbl[3])) - max(rectangle[1], int(lbl[1])))
                        # 분모 계산 = 라벨과 예측 bbox의 합집합 넓이(= 라벨bbox넓이 + 예측bbox넓이 - 교집합넓이)
                        denominator = (int(lbl[2]) * int(lbl[3])) + (
                                    (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])) - numerator
                        # IoU = 교집합넓이 / 합집합넓이
                        iou = numerator / denominator

                    # IoU 가 0.5 이상이면 올바르게 예측했다고 판단
                    # PR 계산을 위해 Confidence, IoU, TP여부, FP여부 저장 및 해당 bbox 제거(중복 집계 방지)
                    if iou >= 0.5:
                        confidences.append(rectangle[4])
                        ious.append(iou)
                        true_positives.append(1)
                        false_positives.append(0)
                        rectangles.remove(rectangle)
                        break
            except:
                continue

        # FP 케이스는 위 조건문에서 집계되지 않으므로 따로 추가해줌, FP 케이스인지는 모든 라벨을 검토한 후에 확인 가능하기 때문
        try:
            for rectangle in rectangles:
                confidences.append(rectangle[4])
                ious.append(0)
                true_positives.append(0)
                false_positives.append(1)
        except:
            pass

    print("num_of_object : {}".format(num_of_object))
    print("input rectangles: {}".format(len(rectangles)))
    result_list = []
    result_list.append(confidences)
    result_list.append(ious)
    result_list.append(true_positives)
    result_list.append(false_positives)
    result_list = np.transpose(result_list)
    print("result_list : \nconfidence\tiou\tTP\tFP\n{}".format(result_list))

    return result_list, num_of_object


def train(dataset):
    # celebA ===========================================================================================================
    if (dataset == 'c' or dataset == 'C'):
        bbox_labels = []
        label_file = open("list_bbox_celeba.txt", "r")  # celebA 데이터셋에서 제공되는 Bbox 라벨
        label_file.readline()  # 첫번째 줄은 전체 데이터 개수가 써 있음
        label_file.readline()  # 두번째 줄은 헤더가 써 있음

        while True:
            line = label_file.readline()
            if not line: break
            bbox_labels.append(line.split())

        root_dir = './img_celeba'
        file_names = os.walk(root_dir).__next__()[2]
        file_names.sort()
    # ==================================================================================================================

    # Wider ============================================================================================================
    if dataset == 'w' or dataset == 'W':
        bbox_labels = []
        label_file = open('./wider_face_split/wider_face_train_bbx_gt.txt', 'r')
        file_names = []
        while True:
            name = label_file.readline()
            if not name: break
            num = int(label_file.readline().split()[0])
            labels = []
            if num <= 0:
                label = [int(x) for x in label_file.readline().split()]
                labels.append(label)
                pass
            else:
                for i in range(num):
                    label = [int(x) for x in label_file.readline().split()]
                    labels.append(label)

            file_names.append(name.replace("\n", ""))
            bbox_labels.append(labels)

        root_dir = './WIDER_train/images'
    # ==================================================================================================================

    threshold = [0.6, 0.6, 0.7]

    # video_path = 'WalmartArguments
    # _p1.mkv'
    # cap = cv2.VideoCapture(video_path)

    # while (True):
    # ret, img = cap.read()
    pnet_results, pnms_results, rnet_results, rnms_results, onet_results = [], [], [], [], []
    total_objects = 0
    time1_count, time2_count, time3_count, time4_count, time5_count = [], [], [], [], []
    count = 1
    for img_path, bbox_label in zip(file_names, bbox_labels):
        print("image count: {}".format(count))

        count += 1
        img = cv2.imread(os.path.join(root_dir, img_path))

        pnet_result, pnms_result, rnet_result, rnms_result, onet_result, time1, time2, time3, time4, time5, num_of_object\
            = detectFace(img, bbox_label, dataset, threshold)

        pnet_results.extend(pnet_result)
        pnms_results.extend(pnms_result)
        rnet_results.extend(rnet_result)
        rnms_results.extend(rnms_result)
        onet_results.extend(onet_result)

        time1_count.append(time1)
        time2_count.append(time2)
        time3_count.append(time3)
        time4_count.append(time4)
        time5_count.append(time5)

        total_objects += num_of_object

        # draw = img.copy()
        #
        # for rectangle in rectangles:
        #     if rectangle is not None:
        #         W = -int(rectangle[0]) + int(rectangle[2])
        #         H = -int(rectangle[1]) + int(rectangle[3])
        #         paddingH = 0.01 * W
        #         paddingW = 0.02 * H
        #         crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
        #                    int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
        #         try:
        #             crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        #             if crop_img is None:
        #                 continue
        #             if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
        #                 continue
        #             cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
        #                           (255, 0, 0), 1)
        #         except:
        #             continue
        #
        #         # for i in range(5, 15, 2):
        #         #     cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))
        # cv2.imshow("test", draw)
        # c = cv2.waitKey(0) & 0xFF
        # if c == 27 or c == ord('q'):
        #     # break
        #     # return
        #     pass
        #
        # print()
        # # cv2.imwrite('test.jpg', draw)

    # 성능 확인
    # confidence = 알고리즘이 해당 bbox에 대해서 가지는 확신 정도
    # iou = 예측 bbox와 타겟 라벨bbox의 iou(0.5가 넘는 경우 타겟 라벨이라 가정), 타겟 라벨을 찾을 수 없는 경우 0으로 저장됨
    # tp = True positive 여부(0, 1), 예측 bbox의 iou가 0.5를 넘는 경우 올바르게 예측했다고 판단 -> 1 저장, else: 0 저장
    # fp = False positive 여부(0, 1), 예측 bbox가 모든 라벨 bbox와 iou가 0.5를 넘지 못 하는 경우 틀린 예측이라 판단 -> 1 저장, else: 0 저장
    # 즉, TP가 아닌 경우라면 FP임
    pnet_results, pnms_results, rnet_results, rnms_results, onet_results \
        = np.array(pnet_results), np.array(pnms_results), np.array(rnet_results), np.array(rnms_results), np.array(onet_results)

    pnet_confidences, pnet_ious, pnet_TPs, pnet_FPs = pnet_results[:, 0], pnet_results[:, 1], pnet_results[:, 2], pnet_results[:, 3]
    pnms_confidences, pnms_ious, pnms_TPs, pnms_FPs = pnms_results[:, 0], pnms_results[:, 1], pnms_results[:, 2], pnms_results[:, 3]
    rnet_confidences, rnet_ious, rnet_TPs, rnet_FPs = rnet_results[:, 0], rnet_results[:, 1], rnet_results[:, 2], rnet_results[:, 3]
    rnms_confidences, rnms_ious, rnms_TPs, rnms_FPs = rnms_results[:, 0], rnms_results[:, 1], rnms_results[:, 2], rnms_results[:, 3]
    onet_confidences, onet_ious, onet_TPs, onet_FPs = onet_results[:, 0], onet_results[:, 1], onet_results[:, 2], onet_results[:, 3]

    # 이미지 하나 처리마다 시간을 측정, 평균 처리시간으로 fps 계산
    print("pnet fps: {}".format(1 / np.mean(time1_count)))
    print("pnms fps: {}".format(1 / np.mean(time2_count)))
    print("rnet fps: {}".format(1 / np.mean(time3_count)))
    print("rnms fps: {}".format(1 / np.mean(time4_count)))
    print("onet fps: {}".format(1 / np.mean(time5_count)))

    # precision = TP / TP + FP
    pnet_precision = np.sum(pnet_TPs) / (np.sum(pnet_TPs) + np.sum(pnet_FPs))
    pnms_precision = np.sum(pnms_TPs) / (np.sum(pnms_TPs) + np.sum(pnms_FPs))
    rnet_precision = np.sum(rnet_TPs) / (np.sum(rnet_TPs) + np.sum(rnet_FPs))
    rnms_precision = np.sum(rnms_TPs) / (np.sum(rnms_TPs) + np.sum(rnms_FPs))
    onet_precision = np.sum(onet_TPs) / (np.sum(onet_TPs) + np.sum(onet_FPs))

    # recall = TP / TP + FN
    pnet_recall = np.sum(pnet_TPs) / total_objects
    pnms_recall = np.sum(pnms_TPs) / total_objects
    rnet_recall = np.sum(rnet_TPs) / total_objects
    rnms_recall = np.sum(rnms_TPs) / total_objects
    onet_recall = np.sum(onet_TPs) / total_objects

    pnet_f1_score = (2 * pnet_precision * pnet_recall) / (pnet_precision + pnet_recall)
    pnms_f1_score = (2 * pnms_precision * pnms_recall) / (pnms_precision + pnms_recall)
    rnet_f1_score = (2 * rnet_precision * rnet_recall) / (rnet_precision + rnet_recall)
    rnms_f1_score = (2 * rnms_precision * rnms_recall) / (rnms_precision + rnms_recall)
    onet_f1_score = (2 * onet_precision * onet_recall) / (onet_precision + onet_recall)

    print("total_objects: {}".format(total_objects))

    print("pnet_Precision: {}\tRecall: {}\tF1_score: {}".format(pnet_precision, pnet_recall, pnet_f1_score))
    print("pnms_Precision: {}\tRecall: {}\tF1_score: {}".format(pnms_precision, pnms_recall, pnms_f1_score))
    print("rnet_Precision: {}\tRecall: {}\tF1_score: {}".format(rnet_precision, rnet_recall, rnet_f1_score))
    print("rnms_Precision: {}\tRecall: {}\tF1_score: {}".format(rnms_precision, rnms_recall, rnms_f1_score))
    print("onet_Precision: {}\tRecall: {}\tF1_score: {}".format(onet_precision, onet_recall, onet_f1_score))

    # 결과 array, 파일로 저장
    with open('pnet_.npy', 'wb') as f:
        np.save(f, pnet_results)
    with open('pnms_.npy', 'wb') as f:
        np.save(f, pnms_results)
    with open('rnet_.npy', 'wb') as f:
        np.save(f, rnet_results)
    with open('rnms_.npy', 'wb') as f:
        np.save(f, rnms_results)
    with open('onet_.npy', 'wb') as f:
        np.save(f, onet_results)

    with open('pnet_.txt', 'wb') as f:
        np.savetxt(f, pnet_results)
    with open('pnms_.txt', 'wb') as f:
        np.savetxt(f, pnms_results)
    with open('rnet_.txt', 'wb') as f:
        np.savetxt(f, rnet_results)
    with open('rnms_.txt', 'wb') as f:
        np.savetxt(f, rnms_results)
    with open('onet_.txt', 'wb') as f:
        np.savetxt(f, onet_results)


if __name__ == "__main__":
    ds = input("which dataset?(c:celebA, w:wider) >> ")
    train(ds)
