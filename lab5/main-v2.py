from multiprocessing.sharedctypes import Value
import cv2
import numpy as np
import time
from stats import Stats
from utils import drawBoundingBox, drawStatistics, printStatistics, Points, drawlines
import pandas as pd

akaze_thresh:float = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 100 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames

#file = "yml/calibrationparameterRobot.yml"
file = "yml/calibMobile.yml"
calib = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
matCam = calib.getNode("mtx").mat()
dst = calib.getNode("dist").mat()
calib.release()

class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

    def setFirstFrame(self, frame, bb, title:str):
        iSize = len(bb)
        stat = Stats()
        ptContain = np.zeros((iSize, 2))
        i = 0
        for b in bb:
            #ptMask[i] = (b[0], b[1])
            ptContain[i, 0] = b[0]
            ptContain[i, 1] = b[1]
            i += 1
        
        self.first_frame = frame.copy()
        matMask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(matMask, np.int32([ptContain]), (255,0,0))

        # cannot use in ORB
        # self.first_kp, self.first_desc = self.detector.detectAndCompute(self.first_frame, matMask)

        # find the keypoints with ORB
        kp = self.detector.detect(self.first_frame,None)
        # compute the descriptors with ORB
        self.first_kp, self.first_desc = self.detector.compute(self.first_frame, kp)

        # print(self.first_kp[0].pt[0])
        # print(self.first_kp[0].pt[1])
        # print(self.first_kp[0].angle)
        # print(self.first_kp[0].size)
        res = cv2.drawKeypoints(self.first_frame, self.first_kp, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        stat.keypoints = len(self.first_kp)
        drawBoundingBox(self.first_frame, bb)
        cv2.namedWindow("key points of {0}".format(title), cv2.WINDOW_NORMAL);
        cv2.imshow("key points of {0}".format(title), res)
        cv2.waitKey(0)
        cv2.destroyWindow("key points of {0}".format(title))

        cv2.putText(self.first_frame, title, (0, 60), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 4)
        self.object_bb = bb
        return stat

    def process(self, frame):
        stat = Stats()
        start_time = time.time()
        kp, desc = self.detector.detectAndCompute(frame, None)
        stat.keypoints = len(kp)
        matches = self.matcher.knnMatch(self.first_desc, desc, k=2)

        matched1 = []
        matched2 = []
        matched1_keypoints = []
        matched2_keypoints = []
        good = []

        for i,(m,n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good.append(m)
                matched1_keypoints.append(self.first_kp[matches[i][0].queryIdx])
                matched2_keypoints.append(kp[matches[i][0].trainIdx])

        matched1 = np.float32([ self.first_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        matched2 = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        stat.matches = len(matched1)
        homography = None
        if (len(matched1) >= 4):
            homography, inlier_mask = cv2.findHomography(matched1, matched2, cv2.RANSAC, ransac_thresh)
        dt = time.time() - start_time
        stat.fps = 1. / dt
        if (len(matched1) < 4 or homography is None):
            res = cv2.hconcat([self.first_frame, frame])
            stat.inliers = 0
            stat.ratio = 0
            return res, stat
        inliers1 = []
        inliers2 = []
        inliers1_keypoints = []
        inliers2_keypoints = []
        for i in range(len(good)):
            if (inlier_mask[i] > 0):
                new_i = len(inliers1)
                inliers1.append(matched1[i])
                inliers2.append(matched2[i])
                inliers1_keypoints.append(matched1_keypoints[i])
                inliers2_keypoints.append(matched2_keypoints[i])
        inlier_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(inliers1))]
        inliers1 = np.array(inliers1, dtype=np.float32)
        inliers2 = np.array(inliers2, dtype=np.float32)

        stat.inliers = len(inliers1)
        stat.ratio = stat.inliers * 1.0 / stat.matches
        bb = np.array([self.object_bb], dtype=np.float32)
        new_bb = cv2.perspectiveTransform(bb, homography)
        frame_with_bb = frame.copy()
        if (stat.inliers >= bb_min_inliers):
            drawBoundingBox(frame_with_bb, new_bb[0])  
        
        res = cv2.drawMatches(self.first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints, inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        
        return res, stat, inliers1, inliers2

    def getDetector(self):
        return self.detector

def main(method):
    img1_name = 'media/robot-frame1.png'
    img2_name = 'media/robot-frame2.png'
    frame1 = cv2.imread(img1_name)
    frame2 = cv2.imread(img2_name)

    stats = Stats()
    if method == 'akaze':
        tech = cv2.AKAZE_create()
        tech.setThreshold(akaze_thresh)
    elif method == 'orb':
        tech = cv2.ORB_create()

    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
    tracker = Tracker(tech, matcher)

    print("Select a ROI and then press SPACE or ENTER button!")
    print("Cancel the selection process by pressing c button!")
    cv2.namedWindow(img1_name, cv2.WINDOW_NORMAL)
    w1, h1, ch = frame1.shape
    cv2.resizeWindow(img1_name, (h1, w1))
    uBox = cv2.selectROI(img1_name, frame1)
    cv2.destroyAllWindows()
    bb = []
    bb.append((uBox[0], uBox[1]))
    bb.append((uBox[0] + uBox[2], uBox[0] ))
    bb.append((uBox[0] + uBox[2], uBox[0] + uBox[3]))
    bb.append((uBox[0], uBox[0] + uBox[3]))
    frame1_cp = frame1.copy()
    if method == 'akaze':
        stat_a = tracker.setFirstFrame(frame1, bb, "AKAZE")
    elif method == 'orb':
        stat_a = tracker.setFirstFrame(frame1, bb, "ORB")

    draw_stats = stat_a.copy()

    res, stat, inliers1, inliers2 = tracker.process(frame2)
    stats  + stat

    #draw epiline ref https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
    if len(inliers1) >= 5:
        # normalization of the correspondences
        inliers1 = cv2.undistortPoints(inliers1, matCam, dst)
        inliers2 = cv2.undistortPoints(inliers2, matCam, dst)
        inliers1_norm = inliers1.reshape(-1, 2) @np.linalg.inv(matCam[:2, :2]) 
        inliers2_norm = inliers2.reshape(-1, 2) @ np.linalg.inv(matCam[:2, :2])
        # computation of the Essential matrix
        matEss, _ = cv2.findEssentialMat(inliers1_norm, inliers2_norm, matCam, method=cv2.RANSAC) #use of the same threshold
    else:
        raise ValueError("Not enough keypoints to compute Essential Matrx")

    inliers1 = inliers1.reshape(1, -1, 2)
    inliers2 = inliers2.reshape(1, -1, 2)
    new_inliers1, new_inliers2 = cv2.correctMatches(matEss, inliers1, inliers2)
    _, R_est, t_est, _ = cv2.recoverPose(matEss, new_inliers1, new_inliers2)
    print("R:", R_est)
    print("t:", t_est)
    # Check if we need to construct P2 with the SVD or not
    I_0 = np.concatenate((np.eye(3), np.array([0, 0, 0]).reshape(3, 1)), axis=1)
    R_t = np.concatenate((R_est, t_est), axis=1)
    P1 = matCam @ I_0
    P2 = matCam @ R_t
    pts3d_hom = cv2.triangulatePoints(P1, P2, new_inliers1[0].reshape(2, -1), new_inliers2[0].reshape(2, -1))
    pts3d = pts3d_hom[:3, :] / pts3d_hom[3, :]
    df = pd.DataFrame(pts3d.reshape(-1, 3))
    df.to_csv("tmp/pts3d.csv")
    return 0

main('akaze')