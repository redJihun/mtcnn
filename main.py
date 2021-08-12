import os

import numpy
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute, PReLU
from tensorflow.keras.models import Model

import numpy as np

import cv2

import time

import tools


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


def detectFace(img, threshold, Pnet=Pnet(r'12net.h5'), Rnet = Rnet(r'24net.h5')):
    caffe_img = (img.copy() - 127.5) / 127.5
    origin_h, origin_w, ch = caffe_img.shape
    scales = tools.calculateScales(img)
    # scales = [1, 0.5]
    out = []
    #
    t0 = time.time()
    # del scales[:4]

    for scale in scales:
        hs = int(origin_h * scale)
        ws = int(origin_w * scale)
        scale_img = cv2.resize(caffe_img, (ws, hs))
        input = scale_img.reshape(1, *scale_img.shape)
        ouput = Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
        out.append(ouput)

    image_num = len(scales)
    rectangles = []

    for i in range(image_num):
        cls_prob = out[i][0][0][:, :, 1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
        roi = out[i][1][0]
        out_h, out_w = cls_prob.shape
        out_side = max(out_h, out_w)
        # print('calculating img scale #:', i)
        cls_prob = np.swapaxes(cls_prob, 0, 1)
        roi = np.swapaxes(roi, 0, 2)
        rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles, 0.7, 'iou')

    t1 = time.time()
    # print('time for 12 net is: ', t1-t0)

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

    out = Rnet.predict(predict_24_batch)

    cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
    cls_prob = np.array(cls_prob)  # convert to numpy
    roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
    roi_prob = np.array(roi_prob)
    rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
    t2 = time.time()
    # print('time for 24 net is: ', t2-t1)

    # if len(rectangles) == 0:
    return rectangles


    # crop_number = 0
    # predict_batch = []
    # for rectangle in rectangles:
    #     # print('calculating net 48 crop_number:', crop_number)
    #     crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
    #     scale_img = cv2.resize(crop_img, (48, 48))
    #     predict_batch.append(scale_img)
    #     crop_number += 1
    #
    # predict_batch = np.array(predict_batch)
    #
    # output = Onet.predict(predict_batch)
    # cls_prob = output[0]
    # roi_prob = output[1]
    # pts_prob = output[2]  # index
    # # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
    # #                                             threshold[2])
    # rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    # t3 = time.time()
    # print ('time for 48 net is: ', t3-t2)
    #
    # return rectangles


def calculate_iou(rects, bbox_label, ds):
    confidences = []
    ious = []
    true_positives = []
    false_positives = []
    total_objects = len(bbox_label)
    rectangles = rects.copy()
    print("total_objects : {}".format(total_objects))
    print("input rectangles: {}".format(len(rectangles)))
    if ds == 'c':
        for rectangle in rectangles:
            # bbox가 라벨box의 범위를 아예 벗어나는 경우 iou를 계산하지 않음
            if (min(rectangle[2], int(bbox_label[1]) + int(bbox_label[3])) - max(rectangle[0], int(bbox_label[1]))) < 0 or \
                    (min(rectangle[3], int(bbox_label[2])+int(bbox_label[4])) - max(rectangle[1], int(bbox_label[2]))) < 0:
                continue

            numerator = (min(rectangle[2], int(bbox_label[1])+int(bbox_label[3])) - max(rectangle[0], int(bbox_label[1]))) * \
                        (min(rectangle[3], int(bbox_label[2])+int(bbox_label[4])) - max(rectangle[1], int(bbox_label[2])))
            denominator = (int(bbox_label[3]) * int(bbox_label[4])) + ((rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])) - numerator

            iou = numerator / denominator
            ious.append(iou)
    else:
        for lbl in bbox_label:
            # print("lbl: {}".format(lbl))
            try:
                for rectangle in rectangles:
                    # print("single rectangle: {}".format(rectangle))
                    # bbox가 라벨box의 범위를 아예 벗어나는 경우 iou를 계산하지 않음
                    if (min(rectangle[2], int(lbl[0]) + int(lbl[2])) - max(rectangle[0], int(lbl[0]))) < 0 or \
                            (min(rectangle[3], int(lbl[1]) + int(lbl[3])) - max(rectangle[1], int(lbl[1]))) < 0:
                        continue

                    else:
                        # 분자 계산 = 라벨과 예측 bbox의 교집합 넓이
                        numerator = (min(rectangle[2], int(lbl[0]) + int(lbl[2])) - max(rectangle[0], int(lbl[0]))) * \
                                    (min(rectangle[3], int(lbl[1]) + int(lbl[3])) - max(rectangle[1], int(lbl[1])))
                        # 분모 계산 = 라벨과 예측 bbox의 합집합 넓이(= 라벨bbox넓이 + 예측bbox넓이 - 교집합넓이)
                        denominator = (int(lbl[2]) * int(lbl[3])) + ((rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])) - numerator
                        # IoU = 교집합넓이 / 합집합넓이
                        iou = numerator / denominator
                        # print("iou: {}".format(iou))

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
    # confidences = np.reshape((None, 1), confidences)
    result_list = []
    result_list.append(confidences)
    result_list.append(ious)
    result_list.append(true_positives)
    result_list.append(false_positives)
    result_list = np.transpose(result_list)
    print("result_list : \nconfidence\tiou\tTP\tFP\n{}".format(result_list))
    # result_list = np.concatenate((np.reshape((-1, 1), confidences), np.reshape((-1, 1), ious), np.reshape((-1, 1), true_positives), np.reshape((-1, 1), false_positives)), axis=1)

    return result_list, total_objects


def train(dataset):
    # celebA ===========================================================================================================
    if(dataset=='c' or dataset=='C'):
        bbox_labels = []
        label_file = open("list_bbox_celeba.txt", "r")          # celebA 데이터셋에서 제공되는 Bbox 라벨
        label_file.readline()           # 첫번째 줄은 전체 데이터 개수가 써 있음
        label_file.readline()           # 두번째 줄은 헤더가 써 있음

        while True:
            line = label_file.readline()
            if not line: break
            bbox_labels.append(line.split())

        root_dir = './img_celeba'
        file_names = os.walk(root_dir).__next__()[2]
        # file_names = []
        # for path in img_paths:
        #     file_names.append(os.path.join(root_dir, path))
        file_names.sort()
    # ==================================================================================================================

    # Wider ============================================================================================================
    if dataset=='w' or dataset=='W':
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
        # imgs = []
        # for path in file_names:
        #     imgs.append(os.path.join(root_dir, path))
    # ==================================================================================================================

    threshold = [0.6, 0.6, 0.7]
    # video_path = 'WalmartArguments
    # _p1.mkv'
    # cap = cv2.VideoCapture(video_path)

    # while (True):
    # ret, img = cap.read()
    results = []
    total_objects = 0
    time_count = []
    count = 1
    for img_path, bbox_label in zip(file_names, bbox_labels):
        print("image count: {}".format(count))
        t1 = time.time()
        count += 1
        img = cv2.imread(os.path.join(root_dir, img_path))

        rectangles = detectFace(img, threshold)

        result, num_of_object = calculate_iou(rectangles, bbox_label, dataset)
        results.append(result)
        total_objects += num_of_object
        # for iou in ious:
        #     iou_list.append(iou)
        #     if iou > 0.5:
        #         accuracy.append(1)
        #         tp_list.append(iou)
        #     else:
        #         accuracy.append(0)

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
        # # print("mean of IoU = {}".format(np.mean(iou_list)))
        # # print("accuracy = {}".format(np.mean(accuracy)))
        # # print("mean of TP IoU = {}".format(np.mean(tp_list)))
        # cv2.imshow("test", draw)
        # c = cv2.waitKey(0) & 0xFF
        # if c == 27 or c == ord('q'):
        #     # break
        #     # return
        #     pass
        # print()

        # cv2.imwrite('test.jpg', draw)
        t2 = time.time()
        time_count.append(t2-t1)

    confidences, ious, tps, fps = results[:][0], results[:][1], results[:][2], results[:][3]
    print("fps: {}".format(60/np.mean(time_count)))
    precision, recall = np.sum(tps) / (np.sum(tps) + np.sum(fps)), np.sum(tps) / (np.sum(tps) + total_objects - np.sum(tps))
    print("Precision: {}\tRecall: {}\tF1_score: {}".format(
        precision,
        recall,
        (2*precision*recall) / (precision+recall)
    ))
    print(np.mean(results[:][2]))
    numpy.savetxt('result.txt', results)


if __name__ == "__main__":
    ds = input("which dataset?(c:celebA, w:wider) >> ")
    train(ds)

