﻿from argparse import RawDescriptionHelpFormatter
from   pathlib              import Path
import unittest
import time
import cv2
import numpy as np
import sys
import math
import os

class Tracker(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.5f minutes' % (self.id(), (t/60)))

    def testOne(self):

        print('-'*100 + '\n' + ' '*50 + 'Running Tracker:' + '\n' + '-'*100)

        self.setParameters()

        self.runTracker()

    #---------------------------------------------------------------------------------------------------------------------

    def setParameters(self):

        # short video using this: ffmpeg -i ENY_30_335.mp4 -ss 00:05:30 -t 00:01:20 -c:v copy -c:a copy ENY_30_335_len_1m2s.mp4

        # basePath = r"/home/yossi/Documents/database/hadar"
        basePath = r"C:\Users\User\Documents\dataBase\DSP-IP"

        # relPath = r"videos/IR_Videos/drone1_1m.mp4"
        relPath = r"videos/OpticalTracker\Attack\Attack\AVT_FOV Fixed FOV 8_Russian_motion.avi"

        outRelPath = r"videos/OpticalTracker\outputVideos/output_AVT_FOV Fixed FOV 8_Russian_motion.mp4"

        videoPath = os.path.join(basePath , relPath)      
        self.outputVideoPath = os.path.join(basePath , outRelPath)

        print(f"\n{videoPath=}")
        print(f"{self.outputVideoPath=}\n")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 0.9 # 0.9 for my laptop smaller screen

        self.maxNumOfFeaturesShown = 1 # number of circles annotating features (corners) found in the image

        self.initialFeatureThreshold = 0.001 # lower myShiTomasiNewFeaturesThreshold to find the first feature
        self.myShiTomasiNewFeaturesThreshold = 0.01 # Remove values < 0.01. cv2.cornerMinEigenVal outputs value=min(λ1,λ2) for each pixel.
        self.opticalFlowErrorThreshold = 10 # Removes error > Threshold. cv2.calcOpticalFlowPyrLK outputs position (p1), status (st), and error (er). 
        
        self.color = np.random.randint(0, 255, (self.maxNumOfFeaturesShown, 3))             # Create some random colors for corner annotaion

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        firstFrame, p0 = self.getFirstImage()

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(self.frameWidth,self.frameHeight))
	
        self.startTrakcerLoop(firstFrame, p0)

    #---------------------------------------------------------------------------------------------------------------------

    def getFirstImage(self):

        frameNumber = 0
        _, firstFrame = self.cap.read()
        while(frameNumber < self.startDetectionFrame): # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1

        if self.screenSizeFactor != 1: # for my laptop smaller screen
            firstFrame = cv2.resize(firstFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

        self.frameWidth, self.frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        xi, yi, self.bboxWidth, self.bboxHeight = cv2.selectROI(firstFrame)

        self.x0, self.x1 = xi, xi + self.bboxWidth
        self.y0, self.y1 = yi, yi + self.bboxHeight

        self.linesMask  = np.zeros_like((firstFrame))
        self.roiMask    = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
        self.roiMask[self.y0:self.y1, self.x0:self.x1] = 255

        goodFeatures = self.getNewFeatures(firstFrame, Threshold=self.initialFeatureThreshold)
        p0 = np.array(goodFeatures).copy().astype(np.float32)

        self.drawFeatures(firstFrame, newFeatures = p0, oldFeatures = p0)

        cv2.imshow('frame', firstFrame) 
        
        return firstFrame, p0
        
    #---------------------------------------------------------------------------------------------------------------------

    def startTrakcerLoop(self, firstFrame, p0):
        oldFrame = firstFrame
        
        while(True):
            _, newFrame = self.cap.read()
            if type(newFrame) == type(None):
                break
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            p1Good = self.getOpticalFlowFeatures(oldFrame, newFrame, p0)

            # print(f"{p1Good=}")
            if p1Good.size > 0:
                # print(f"features found, in {p1Good=}")
                newFrame = self.drawFeatures(frame=newFrame, newFeatures=p1Good, oldFeatures=p0)
                self.shiftRoiMask(p1Good)
            else:
                # print(f"no features found, looking for features in {self.x0=}, {self.x1=}, {self.y0=}, {self.y1=}")
                p1Good = []
                goodFeatures = self.getNewFeatures(newFrame)
                if len(goodFeatures) > 0:
                    p0 = np.array(goodFeatures).copy().astype(np.float32)
                    self.color = np.random.randint(0, 255, (self.maxNumOfFeaturesShown, 3))             # Create some random colors for corner annotaion

            oldFrame = newFrame.copy()
        
            cv2.rectangle(newFrame, (self.x0, self.y0), (self.x1, self.y1), (0, 255, 0), 1)
            cv2.imshow('frame', newFrame) 
            
            self.outputVideo.write(newFrame)
               
            k = cv2.waitKey(self.waitKeyParameter)
            if k == 27: 
                break

        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------

    def getNewFeatures(self, newFrame, Threshold=None):
        if Threshold == None:
            Threshold = self.myShiTomasiNewFeaturesThreshold
        # This function returns all features with threshold > absolute val
        src = newFrame[self.y0: self.y1, self.x0: self.x1]
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow('src_gray', src_gray) 

        myShiTomasi_dst = cv2.cornerMinEigenVal(src=src_gray, blockSize=3, ksize=3, borderType=cv2.BORDER_REFLECT_101)

        goodFeatures = []
        myShiTomasi_minVal, myShiTomasi_maxVal, _, _ = cv2.minMaxLoc(myShiTomasi_dst)

        for i in range(src_gray.shape[0]):
            for j in range(src_gray.shape[1]):
                if myShiTomasi_dst[i,j] > Threshold: # myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal ) * self.myShiTomasiThreshold:
                    goodFeatures.append([[self.x0 + j, self.y0 + i]])

        return goodFeatures

    #---------------------------------------------------------------------------------------------------------------------

    def shiftRoiMask(self, p1):

        p1X, p1Y = p1[0].ravel().astype(np.uint32)

        self.x0, self.x1 = p1X - self.bboxWidth //2, p1X + self.bboxWidth //2
        self.y0, self.y1 = p1Y - self.bboxHeight//2, p1Y + self.bboxHeight//2

        if self.x0 < 0 or self.x1 > self.frameWidth or self.y0 < 0 or self.y1 > self.frameHeight:
            sys.exit("out of screen!")

        self.roiMask    = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
        self.roiMask[self.y0:self.y1, self.x0:self.x1] = 255

    #---------------------------------------------------------------------------------------------------------------------

    def getOpticalFlowFeatures(self, oldImg, newImg, p0):
        # Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.            
        # Assuming:         
        # 1.The pixel intensities of an object do not change between consecutive frames.
        # 2. Neighbouring pixels have similar motion.

        oldGray = cv2.cvtColor(oldImg, cv2.COLOR_BGR2GRAY)
        newGray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
                prevImg=oldGray,        # first 8-bit input image
                nextImg=newGray,        # second input image
                prevPts=p0,             # vector of 2D points for which the flow needs to be found.
                nextPts=None,
                winSize = (21, 21),       # size of the search window at each pyramid level
                maxLevel = 2,           #0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on                
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), # parameter, specifying the termination criteria of the iterative search algorithm                
                flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold = self.myShiTomasiNewFeaturesThreshold
         )# criteria -> after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon

        p1Good = self.removeBadFeatures(p1, st, err)   

        return p1Good

    #---------------------------------------------------------------------------------------------------------------------

    def removeBadFeatures(self, p1, st, err):

        # print(f"{p1=}")
        # print(f"{st=}")	
        # print(f"{err=}")
    
        status0List = []
        
        for errIdx, errEle in enumerate(err):
            if errEle[0] > self.opticalFlowErrorThreshold or st[errIdx] == [0] or \
             (not self.y0 <= p1[errIdx][0][1] <= self.y1) or (not self.x0 <= p1[errIdx][0][0] <= self.x1):
                status0List.append(errIdx)

        p1Good = np.delete(p1,status0List, axis=0)
        
        return p1Good

    #---------------------------------------------------------------------------------------------------------------------

    def drawFeatures(self, frame, newFeatures, oldFeatures):
         
        for idx, (new, old) in enumerate(zip(newFeatures[0:self.maxNumOfFeaturesShown], oldFeatures[0:self.maxNumOfFeaturesShown])): 
				
            p1X, p1Y = new.ravel().astype(np.uint32)
            p0X, p0Y = old.ravel().astype(np.uint32)

            self.linesMask = cv2.line(self.linesMask, (p1X, p1Y), (p0X, p0Y), self.color[idx].tolist(), 2)

            frame = cv2.circle(frame, (p1X, p1Y), 5, self.color[idx].tolist(), -1)
    
        if self.to_draw_line:
            frame = cv2.add(frame, self.linesMask)
        return frame

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Tracker)
    unittest.TextTestRunner(verbosity=0).run(suite)





                # print(f"{newFeatruePoints=}")
                # print(f"{p0=}")
                # if type(newFeatruePoints) != type(None) and newFeatruePoints.size > p0.size:
                #     print("newFeatruePoints.size > p0.size")
                #     p0 = self.getClosestFeatures(newFeatruePoints, p0)
                # else:
                #     print("p0 same size")
                    
                #     p0 = newFeatruePoints


    # #---------------------------------------------------------------------------------------------------------------------

    # def getClosestFeatures(self, newFeatruePoints, p0):

    #     p0Good = []
    #     for oldP0 in p0:

    #         minDistance = np.Inf
    #         for newIdx, newP0 in enumerate(newFeatruePoints):

    #             dist = np.linalg.norm(newP0-oldP0)
    #             if dist < minDistance:
    #                 minDistanceIdx = newIdx
    #         p0Good.append(newFeatruePoints[minDistanceIdx])
    #     p0Good = np.array(p0Good)

    #     return p0Good




    #---------------------------------------------------------------------------------------------------------------------

    # def searchLargerRoi(self, newFrame):
    #     newFeatures = None
    #     span = 1
    #     while type(newFeatures) == type(None):
    #         span *= 2
    #         roiMask = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
    #         roiMask[self.y0 - span:self.y1 + span, self.x0 - span:self.x1 + span] = 255

    #         newFeatures = self.getImageFeatures(newFrame, roiMask)

    #     return newFeatures




    #---------------------------------------------------------------------------------------------------------------------
'''         cv2.goodFeaturesToTrack uses the min eigen value and takes qualityLevel*maxValue rather than above a threshold as implemented above

    def getImageFeatures(self, img, roiMask = None):
        # This function returns all features with threshold > (qualityLevel* max_value_feature), rather than absolute val as threshold
        #https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
        # goodFeaturesToTrack 2 options: Shi-Tomasi or Harris corner score
        # By default using Shi-Tomasi algorithm (useHarrisDetector = False), based on Harris corner detection.
        if type(roiMask) != type(None):
            roiMask = roiMask
        else:
            roiMask = self.roiMask

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(   # Outputs vector of detected corners.
            image = imgGray,            # Input 8-bit or floating-point 32-bit, single-channel image.              
            maxCorners = 100,           # Maximum number of corners to return (-1 for all). If there are more corners than are found, the strongest of them is returned.
            qualityLevel = 0.3,         # minimal accepted quality of image corners. qualityLevel is multiplied by max(entire image R): Shi-Tomasi or Harris, depending on useHarrisDetector
            minDistance = 7,           # Minimum possible Euclidean distance between the returned corners.
            mask = roiMask,                # Optional region of interest. It specifies the region in which the corners are detected.
            blockSize = 7,             # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
            useHarrisDetector = False,  # False - use cornerMinEigenVal, True - use cornerHarris
            k = None,                   # 0.04 Free parameter of the Harris detector.relevant only when useHarrisDetector = True
            )

        return p0

        # cv2.cornerMinEigenVal and cv2.cornerHarris works similary:
        # For every pixel p in an image, the functions considers a blockSize × blockSize neighborhood S(p)
        # It calculates the covariation matrix of derivatives over the neighborhood as 2x2 matrix M, and from M calcultes a score R.
        # Shi-Tomasi score R=min(λ1,λ2) using cv2.cornerMinEigenVal and Harris function response score R=det(M)−k(trace(M))2 using cv2.cornerHarris
        # For each pixel in the image: cv2.cornerMinEigenVal retun R=the minimal eigen value and cv2.cornerHarris returns R=det(M)−k(trace(M))2

        # goodFeaturesToTrack takes cv2.cornerMinEigenVal and cv2.cornerHarris results image (image with pixels as the results R).
        # Performs a non-maximum suppression (the local maximums in 3 x 3 neighborhood are retained) over the image and finds the corner with the maximum value
        # Then, thresholds the corners (the R value) with (qualityLevel *  maximum value). corners less than this value are rejected.
        # The remaining corners are sorted by the quality measure in the descending order.
        # Function throws away each corner for which there is a stronger corner at a distance less than maxDistance.

'''