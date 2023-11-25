from argparse import RawDescriptionHelpFormatter
from   pathlib              import Path
import unittest
import time
import cv2
import numpy as np
import sys
import math
import os

class ObjectDetection(unittest.TestCase):
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
               
        # basePath = r"/home/yossi/Documents/database/hadar"
        basePath = r"C:\Users\User\Documents\dataBase\DSP-IP"

        relPath = r"videos\IR_Videos\IR_AIRPLANE_002.mp4"
        outRelPath = r"videos\IR_Videos\outputVideos\IR_AIRPLANE_002.mp4"

        videoPath = os.path.join(basePath , relPath) 
        self.outputVideoPath = os.path.join(basePath , outRelPath)
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.frameCutFactor = 0.9 # to cut annotation like FLIR and other trademarks...
        
        self.bboxWidth, self.bboxHeight = 10, 10

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.startTrakcerLoop()

        cv2.destroyAllWindows() 
        self.cap.release()
        
    #---------------------------------------------------------------------------------------------------------------------

    def startTrakcerLoop(self):
        frameNumber = 0
        _, firstFrame = self.cap.read()
        while(frameNumber < self.startDetectionFrame): # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1
            
        frameWidth  = firstFrame.shape[1] - 2 * (firstFrame.shape[1] - int(firstFrame.shape[1] * self.frameCutFactor))
        frameHeight = firstFrame.shape[0] - 2 * (firstFrame.shape[0] - int(firstFrame.shape[0] * self.frameCutFactor))

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(frameWidth, frameHeight))

        while(True):
            _, newFrame = self.cap.read()
            if type(newFrame) == type(None):
                break
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            self.frameWidth, self.frameHeight = int(newFrame.shape[1] * self.frameCutFactor), int(newFrame.shape[0] * self.frameCutFactor)

            newFrame = newFrame[newFrame.shape[0] - self.frameHeight : self.frameHeight,\
                                newFrame.shape[1] - self.frameWidth  : self.frameWidth ]
            print(f"{newFrame.shape=}")

            self.findBbox(newFrame)

            cv2.imshow('frame', newFrame) 

            self.outputVideo.write(newFrame)
               
            k = cv2.waitKey(self.waitKeyParameter)
            if k == 27: 
                break

        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------
        
    def findBbox(self, frame):
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bluredImage = cv2.GaussianBlur(src=frameGray, ksize=(5, 5), sigmaX=3)

        ret, thresholdImage = cv2.threshold(bluredImage, 187, 255, cv2.THRESH_BINARY)

        # morphImage =  cv2.morphologyEx(src=thresholdImage, op=cv2.MORPH_OPEN, kernel=(11,11), iterations=1)

        # cv2.imshow("morphImage", thresholdImage)
        # cv2.waitKey(0)

        numLabels, labels, stats, centroids =  cv2.connectedComponentsWithStats(image=thresholdImage, connectivity=4, ltype=cv2.CV_32S)
	    # x,y,w,h,area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if numLabels > 1:
            areasList = stats[:, cv2.CC_STAT_AREA] 

            print(f"{stats=}")
            print(f"{areasList=}")
            maxAreaIdx = np.argmax(areasList[1:]) # index 0 is for background with the maximal area
            print(f"{maxAreaIdx=}")
            self.cX,self.cY = centroids[1:][maxAreaIdx]
            print(f"{self.cX,self.cY}")

            # finalImage = morphImage.copy()
    
            # (yi, xi) = np.unravel_index(finalImage.argmax(), finalImage.shape)

            # print(f"{xi=}, {yi=}")
            # x0, x1 = xi-self.bboxWidth,  xi+self.bboxWidth
            # y0, y1 = yi-self.bboxHeight, yi+self.bboxHeight

            # cv2.rectangle(img=frame, pt1=(x0, y0), pt2=(x1, y1), color=(255,0,0), thickness=1)   
        cv2.circle(img=frame, center=(int(self.cX), int(self.cY)), radius=1, color=(0,0,255), thickness=-1)   

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)