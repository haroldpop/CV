import enum
import cv2
from cv2 import split
import numpy as np
import os

from homography import Homography

VIDEO_FILE = "/home/pop/Documents/School/CV/code/labs/lab4/Lab01-robot-video.mp4"
frame_dir = "/home/pop/Documents/School/CV/code/labs/lab4/frame"
maskframe_dir = "/home/pop/Documents/School/CV/code/labs/lab4/maskframe"


def getFloorPixel(frame, c1, c2):
    floor_pixel = []
    nofloor_pixel = []
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j] == c1 :
                nofloor_pixel.append((i, j))
            else:
                floor_pixel.append((i, j))
    return floor_pixel, nofloor_pixel


def createBinArray(h_value, s_value, v_value,  nb_bins_h, nb_bins_s, nb_bins_v):
    binsh = [int(k*(255/nb_bins_h)) for k in range(nb_bins_h)]
    binss = [int(k*(255/nb_bins_s)) for k in range(nb_bins_s)]
    binsv = [int(k*(255/nb_bins_v)) for k in range(nb_bins_v)]
    h_inds = np.digitize(h_value, binsh)
    s_inds = np.digitize(s_value, binss)
    v_inds = np.digitize(v_value, binsv)
    return h_inds, s_inds, v_inds


def getAnnotFrame(frame_dir, maskframe_dir):
    annot_frame = []
    for frame in os.listdir(frame_dir):
        if frame in os.listdir(maskframe_dir):
            annot_frame.append(frame)
    return annot_frame


def split_train_test(annot_frame, test_size):
    annot_len = len(annot_frame)
    test_len = int(annot_len*test_size)
    annot_frame_cp = np.copy(annot_frame)
    np.random.shuffle(annot_frame_cp)
    test_set = annot_frame_cp[:test_len]
    train_set = annot_frame_cp[test_len:]
    print('NB of training images {}.    Nb of testing image {}'.format(len(train_set), len(test_set)))
    return train_set, test_set


def getProb(train_set, nb_bins_h, nb_bins_s, nb_bins_v):
    logprobTablehsfloor = np.zeros((nb_bins_h, nb_bins_s, nb_bins_v))
    logprobTablehsnofloor = np.zeros((nb_bins_h, nb_bins_s, nb_bins_v))
    pfloor = 1
    pnofloor = 1

    for frame in train_set:
        frame_image_path = frame_dir + '/' + frame
        frame_image = cv2.imread(frame_image_path, cv2.IMREAD_COLOR)

        frame_hsv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
        
        maskframe_image_path = maskframe_dir + '/' + frame
        maskframe_image = cv2.imread(maskframe_image_path, cv2.IMREAD_GRAYSCALE)
        
        flat_mask = maskframe_image.reshape(-1, 1)
        c1, c2 = np.unique(flat_mask)
        #if the first color is not black we put it black
        if c1 != 0:
            aux = c2
            c2 = c1
            c1 = aux

        floor_pixel, nofloor_pixel = getFloorPixel(maskframe_image, c1, c2)
        pfloor *= len(floor_pixel) / (len(floor_pixel) + len(nofloor_pixel))
        pnofloor *= len(nofloor_pixel) / (len(floor_pixel) + len(nofloor_pixel))
        

        hsvfloor = frame_hsv[tuple(np.transpose(floor_pixel))]
        hfloor = hsvfloor[:, 0].reshape(-1, 1)
        sfloor = hsvfloor[:, 1].reshape(-1, 1)
        vfloor = hsvfloor[:, 2].reshape(-1, 1)
        h_indfloor, s_indfloor, v_indfloor = createBinArray(hfloor, sfloor, vfloor, nb_bins_h, nb_bins_s, nb_bins_v)
        totfloor = h_indfloor.shape[0]

        hsvnofloor = frame_hsv[tuple(np.transpose(nofloor_pixel))]
        hnofloor = hsvnofloor[:, 0].reshape(-1, 1)
        snofloor = hsvnofloor[:, 1].reshape(-1, 1)
        vnofloor = hsvnofloor[:, 2].reshape(-1, 1)
        h_indnofloor, s_indnofloor, v_indnofloor = createBinArray(hnofloor, snofloor, vnofloor, nb_bins_h, nb_bins_s, nb_bins_v)
        totnofloor = h_indnofloor.shape[0]

        for i in range(nb_bins_h):
            for j in range(nb_bins_s):
                for k in range(nb_bins_v):
                    interfloor = np.where(h_indfloor==i)[0].shape[0] + np.where(s_indfloor==j)[0].shape[0] + np.where(v_indfloor==k)[0].shape[0]
                    logprobTablehsfloor[i][j][k] += np.log(interfloor / totfloor)

                    internofloor = np.where(h_indnofloor==i)[0].shape[0] + np.where(s_indnofloor==j)[0].shape[0] + np.where(v_indnofloor==k)[0].shape[0]
                    logprobTablehsnofloor[i][j][k] += np.log(internofloor / totnofloor)

    probTablehsfloor = np.exp(logprobTablehsfloor)
    probTablehsnofloor = np.exp(logprobTablehsnofloor)
    return pfloor, pnofloor, probTablehsfloor, probTablehsnofloor


def generateMask(image, nb_bins_h, nb_bins_s, nb_bins_v, pfloor, pnofloor, phsf, phsnf):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_value = image_hsv[:, :, 0].reshape(-1, 1)
    s_value = image_hsv[:, :, 1].reshape(-1, 1)
    v_value = image_hsv[:, :, 2].reshape(-1, 1)
    h_inds, s_inds, v_inds = createBinArray(h_value, s_value, v_value, nb_bins_h, nb_bins_s, nb_bins_v)
    #rescale to make it start at 0
    h_inds = h_inds - 1
    s_inds = s_inds -1
    v_inds = v_inds - 1
    generate_mask = np.zeros(h_inds.shape, dtype=np.uint8)
    for i in range(h_inds.shape[0]):
        idx_h = int(h_inds[i])
        idx_s = int(s_inds[i])
        idx_v = int(v_inds[i])
        p_hsv_floor = phsf[idx_h][idx_s][idx_v]
        p_hsv_nofloor = phsnf[idx_h][idx_s][idx_v]
        if p_hsv_floor*pfloor >  p_hsv_nofloor*pnofloor and p_hsv_floor > 0.001:
            generate_mask[i] = 255  #check the color
    generate_mask = generate_mask.reshape(image.shape[:2])
    return generate_mask


def displayMaskImage(frame, nb_bins, pfloor, pnofloor, phsf, phsnf):
    """
    frame: path to the image
    """
    if len(nb_bins) != 3 :
        return "Please enter nb_h, nb_s, nb_v in nb_bins as a tuple"
    nb_bins_h, nb_bins_s, nb_bins_v = nb_bins
    image = cv2.imread(frame, cv2.IMREAD_COLOR)
    generate_mask = generateMask(image, nb_bins_h, nb_bins_s, nb_bins_v, pfloor, pnofloor, phsf, phsnf)
    mask_image = cv2.bitwise_and(image, image, mask=generate_mask)
    res = np.concatenate((image, generate_mask, mask_image), axis = 1)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", res)
    cv2.waitKey(0)

def evaluateModel(test_set, nb_bins_h, nb_bins_s, nb_bins_v , pfloor, pnofloor, phsf, phsnf):
    # res = [true_0, true_255, false_0, false_255]
    res = np.zeros((4, 1))

    for frame in test_set:
        frame_image_path = frame_dir + '/' + frame
        frame_image = cv2.imread(frame_image_path, cv2.IMREAD_COLOR)
        generate_mask = generateMask(frame_image, nb_bins_h, nb_bins_s, nb_bins_v, pfloor, pnofloor, phsf, phsnf)

        mask_frame_image_path = maskframe_dir + '/' + frame
        mask_frame_image = cv2.imread(mask_frame_image_path, cv2.IMREAD_COLOR)
        mask_frame_image = cv2.cvtColor(mask_frame_image, cv2.COLOR_BGR2GRAY)
        ret, mask_frame_image = cv2.threshold(mask_frame_image, 10, 255, cv2.THRESH_BINARY)

        pred = generate_mask.reshape(-1, )
        truth = mask_frame_image.reshape(-1, )
        p0_bool = pred == 0
        t0_bool = truth == 0
        p255_bool = pred == 255
        t255_bool = truth == 255
        t0 = np.where(p0_bool*t0_bool == True)[0].shape[0]
        t255 = np.where(p255_bool*t255_bool == True)[0].shape[0]
        f0 = np.where(p0_bool*t255_bool == True)[0].shape[0]
        f255 = np.where(p255_bool*t0_bool == True)[0].shape[0]

        for i, value in enumerate((t0, t255, f0, f255)):
            res[i] += value
        break
    
    #take the mean
    res = res
    #compute indicators
    acc = (res[0] + res[1]) / np.sum(res)
    prec = res[0] / (res[0] + res[2]) #precision computed for the activated mask true means black pixel value of 0
    recall = res[0] / (res[0] + res[3])
    f1 = 2*prec*recall / (recall + prec)

    return acc, prec, recall, f1
    


def crosskfoldEvaluation(k_folds, annot_frame, nb_bins_h, nb_bins_s, nb_bins_v):
    fold_size = int(len(annot_frame) / k_folds)
    annot_cp = np.copy(annot_frame)
    np.random.shuffle(annot_frame)
    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    for fold_idx, i in enumerate(range(0, len(annot_frame), fold_size)):
        test_frame = annot_frame[i:i+fold_size]
        train_frame = annot_frame[:i] + annot_frame[i+fold_size:]
        pfloor, pnofloor, phsf, phsnf = getProb(train_frame, nb_bins_h, nb_bins_s, nb_bins_v)
        acc, prec, recall, f1 = evaluateModel(test_frame, nb_bins_h, nb_bins_s, nb_bins_v, pfloor, pnofloor, phsf, phsnf)
        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)

    acc_mean = np.mean(np.array(acc_list))
    prec_mean = np.mean(np.array(prec_list))
    recall_mean = np.mean(np.array(recall_list))
    f1_mean = np.mean(np.array(f1_list))
    return acc_mean, prec_mean, recall_mean, f1_mean

def saveVideo(videoPath, pfloor, pnofloor, phsf, phsnf):
    nb_bins_h, nb_bins_s, nb_bins_v = phsf.shape
    t_start = time.time()
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        raise ValueError("Error opening video stream or file")

    video = cv2.VideoWriter("Segmentationv1.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, resolution)
    
    cFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iFrame = 0
    while True:
        ret, matFrameCapture = cap.read()
        if not ret:
            break
        matFrameDisplay = matFrameCapture
        generate_mask = generateMask(matFrameDisplay, nb_bins_h, nb_bins_s, nb_bins_v, pfloor, pnofloor, phsf, phsnf)
        mask_image = cv2.bitwise_and(matFrameDisplay, matFrameDisplay, mask=generate_mask)
        matRes = np.concatenate((matFrameDisplay, generate_mask, mask_image), axis = 0)
        matRes = cv2.resize(matRes, resolution, cv2.INTER_CUBIC)
        video.write(matRes)

        iKey = cv2.waitKey(1)
        if iKey == ord('q') or iKey == ord('Q'):
            return 0

        iFrame += 1
        if iFrame % 100 == 0:
            print("Progress: {}/{}".format(iFrame, cFrames))
    
    t_end = time.time()
    print("Time to save vide: {:.3f}".format(t_end-t_start))
    video.release()
    cap.release()

    





import time
resolution = (3600, 2700)
nb_bins_h, nb_bins_s, nb_bins_v  = 64, 16, 16
nb_bins = (nb_bins_h, nb_bins_s, nb_bins_v)
print("Nb bins for h: {} \n Nb bins for s: {} \n Nb bins for v {}".format(nb_bins_h, nb_bins_s, nb_bins_v))

test_size = 0.2
annot_frame = getAnnotFrame(frame_dir, maskframe_dir)
train_frame, test_frame = split_train_test(annot_frame, test_size)

tstart = time.time()
pfloor, pnofloor, phsf, phsnf = getProb(train_frame, nb_bins_h, nb_bins_s, nb_bins_v)
tend = time.time()
print("Training time {:.3f}".format(tend-tstart))
tstart = time.time()

#displayMaskImage(test, nb_bins, pfloor, pnofloor, phsf, phsnf)


# k_folds = 8
# acc, prec, recall, f1 = crosskfoldEvaluation(k_folds, annot_frame, nb_bins_h, nb_bins_s, nb_bins_v)
# print("Final results: \n acc: {}, prec: {}, recall: {}, f1: {}".format(acc, prec, recall, f1))
        
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# 