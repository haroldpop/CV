import numpy as np
import cv2
import glob
import os
import time

from homography import Homography


def getMask(matFrameHSV, aProbFloor, aProbNotFloor, pfloor, pnotfloor):
    h, w, c = matFrameHSV.shape
    hbins, sbins, vbins = aProbFloor.shape
    matMask = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            hsv = matFrameHSV[i][j]
            Hbin = int(hsv[0]*hbins / 256)
            Sbin = int(hsv[1]*sbins / 256)
            Vbin = int(hsv[2]*vbins / 256)
            if aProbFloor[Hbin][Sbin][Vbin]*pfloor > aProbNotFloor[Hbin][Sbin][Vbin]*pnotfloor and aProbFloor[Hbin][Sbin][Vbin] > 0.001:
                matMask[i][j] = 255
    return matMask

def reg(matFrameDisplay, homo):
    matResult = cv2.warpPerspective(matFrameDisplay, homo.matH, (homo.widthOut, homo.heightOut), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    return matResult

def getFloorHist(aProbFloor, aProbNotFloor, test_size):
    # stack original picture
    pathOriginal = "/home/pop/Documents/School/CV/code/labs/lab4/frame"
    frames = os.listdir(pathOriginal)
    listFrame = []
    for frame in frames:
        im = cv2.imread(pathOriginal + '/' + frame)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        listFrame.append(im)

    # stack mask picture 
    pathMask = "/home/pop/Documents/School/CV/code/labs/lab4/maskframe"
    maskframes = os.listdir(pathMask)
    listMaskFrame = []
    for maskframe in maskframes:
        im = cv2.imread(pathMask + '/' + maskframe)
        listMaskFrame.append(im)
    idxs = np.arange(len(listFrame))
    np.random.shuffle(idxs)
    test_len = int(len(listFrame)*test_size)
    idxs_test = idxs[:test_len]
    idxs_train = idxs[test_len:]
    listFrametrain = []
    listMaskFrameTrain = []
    for idx in idxs_train:
        listFrametrain.append(listFrame[idx])
        listMaskFrameTrain.append(listMaskFrame[idx])
    hbins, sbins, vbins = aProbFloor.shape
    if len(listFrametrain) == len(listMaskFrameTrain):
        cFloor = 0
        cNotFloor = 0
        for t in range(len(listFrametrain)):
            matMaskGray = cv2.cvtColor(listMaskFrameTrain[t], cv2.COLOR_BGR2GRAY)
            matHSVFrame = listFrame[t]
            for i in range(matMaskGray.shape[0]):
                for j in range(matMaskGray.shape[1]):
                    bMasked = matMaskGray[i][j] > 0
                    hsv = matHSVFrame[i][j]
                    Hbin = int(hsv[0]*hbins / 256)
                    Sbin = int(hsv[1]*sbins / 256)
                    Vbin = int(hsv[2]*vbins / 256)
                    if bMasked:
                        aProbFloor[Hbin][Sbin][Vbin] += 1
                        cFloor += 1
                    else:
                        aProbNotFloor[Hbin][Sbin][Vbin] += 1
                        cNotFloor += 1
            for i in range(hbins):
                for j in range(sbins):
                    for k in range(vbins):
                        aProbFloor[i][j][k] /= cFloor
                        aProbNotFloor[i][j][k] /= cNotFloor
            
        return aProbFloor, aProbNotFloor, cFloor, cNotFloor

def maskFrame(matFrame, matMask):
    h, w, c = matFrame.shape
    for i in range(h):
        for j in range(w):
            if matMask[i][j] > 0:
                bgr = matFrame[i][j]
                bgr[1] = 0.5*bgr[1] + 0.5*255
                matFrame[i][j] = bgr
    return matFrame

def main(VIDEO_FILE, homo_file, binsH, binsS, binsV, test_size):

    homo = Homography(homo_file)
    cFloor = 0
    cNotFloor = 0
    aProbFloor = np.zeros((binsH, binsS, binsV))
    aProbNotFloor = np.zeros((binsH, binsS, binsV))

    cap = cv2.VideoCapture(VIDEO_FILE)
    if (cap.isOpened() == False):
        raise ValueError("Error opening video stream or file")

    video = cv2.VideoWriter("Segmentationv2.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, resolution)

    t_start = time.time()
    aProbFloor, aProbNotFloor, cFloor, cNotFloor = getFloorHist(aProbFloor, aProbNotFloor, test_size)
    t_end = time.time()
    print("Training time : {:.3f}s".format(t_end-t_start))

    cFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    iFrame = 0

    t_vid_start = time.time()
    while True:
        ret, matFrameCapture = cap.read()

        if not ret:
            break

        matFrameDisplay = matFrameCapture
        matFrameDisplayHomo = matFrameCapture.copy()

        matFrameHSV = cv2.cvtColor(matFrameDisplay, cv2.COLOR_BGR2HSV)
        matFrameHSVHomo = cv2.cvtColor(matFrameDisplayHomo, cv2.COLOR_BGR2HSV)
        
        pFloor = cFloor / (cFloor + cNotFloor)
        pNotFloor = cNotFloor / (cFloor + cNotFloor)

        matMask = getMask(matFrameHSV, aProbFloor, aProbNotFloor, pFloor, pNotFloor)
        matmaskFrame = maskFrame(matFrameDisplay, matMask)

        matPerspHSVHomo = reg(matFrameHSV, homo)
        matPerspDisplayHomo = reg(matFrameDisplay, homo)

        matPerspMask = getMask(matPerspHSVHomo, aProbFloor, aProbNotFloor, pFloor, pNotFloor)
        matMaskPersp = maskFrame(matPerspDisplayHomo, matPerspMask) 

        matDisp = np.concatenate((matFrameDisplay, matmaskFrame, matMaskPersp), axis=0)
        matDisp = cv2.resize(matDisp, resolution, cv2.INTER_CUBIC)
        video.write(matDisp)

        iKey = cv2.waitKey(1)
        if iKey == ord('q') or iKey == ord('Q'):
            return 0

        iFrame += 1

        if iFrame % 100 == 0:
            print("Progress: {}/{}".format(iFrame, cFrames))

    t_vid_end = time.time()
    print("Time to save the video: {:.3f}".format(t_vid_end-t_vid_start))

    video.release()
    cv2.destroyAllWindows()
    return 0

test_size = 0.2
VIDEO_FILE = "/"
homo_file = "/"
binsH, binsS, binsV = 64, 16, 16
resolution = (3600, 2700)
main(VIDEO_FILE, homo_file, binsH, binsS, binsV, test_size)








    
    

            
