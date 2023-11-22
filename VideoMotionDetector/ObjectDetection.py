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

        videoPath = os.path.join(basePath , relPath) 
        
        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.frameCutFactor = 0.9 # to cut annotation like FLIR and other trademarks...

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.getFirstImage()

        cv2.destroyAllWindows() 
        self.cap.release()

    #---------------------------------------------------------------------------------------------------------------------

    def getFirstImage(self):

        frameNumber = 0
        _, firstFrame = self.cap.read()
        while(frameNumber < self.startDetectionFrame): # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1

        if self.screenSizeFactor != 1: # for my laptop smaller screen
            firstFrame = cv2.resize(firstFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

        self.frameWidth, self.frameHeight = int(firstFrame.shape[1] * self.frameCutFactor), int(firstFrame.shape[0] * self.frameCutFactor)

        firstFrame = firstFrame[firstFrame.shape[0] - self.frameHeight : self.frameHeight,\
                                firstFrame.shape[1] - self.frameWidth  : self.frameWidth ]

        # self.x0, self.x1 = xi, xi + self.bboxWidth
        # self.y0, self.y1 = yi, yi + self.bboxHeight

        # self.linesMask  = np.zeros_like((firstFrame))
        # self.roiMask    = np.zeros((self.frameHeight, self.frameWidth), dtype=np.uint8)
        # self.roiMask[self.y0:self.y1, self.x0:self.x1] = 255

        # goodFeatures = 
        # bluredImage = self.detectObject(firstFrame)
        firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
        firstFrameOrigi = firstFrame.copy()

        bluredImage = cv2.GaussianBlur(src=firstFrame, ksize=(5, 5), sigmaX=3)
        # bluredImage = cv2.medianBlur(src=firstFrame, ksize=5)
        # p0 = np.array(goodFeatures).copy().astype(np.float32)

        # self.drawFeatures(firstFrame, newFeatures = p0, oldFeatures = p0)

        # diffImage = cv2.absdiff(firstFrame, bluredImage)
        # print(f"{diffImage.shape=}")
        # print(f"{type(diffImage)=}")
        ret, thresholdImage = cv2.threshold(bluredImage, 200, 255, cv2.THRESH_BINARY)
        # morphImage =  cv2.erode(bluredImage, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        morphImage =  cv2.morphologyEx(src=thresholdImage, op=cv2.MORPH_OPEN, kernel=(11,11), iterations=1)
        finalImage = morphImage.copy()
   

        (y0, x0) = np.unravel_index(finalImage.argmax(), finalImage.shape)

        print(f"{x0=}, {y0=}")
        bboxWidth, bboxHeight = 10, 10

        cv2.rectangle(img=firstFrame, pt1=(x0-bboxWidth, y0-bboxHeight), pt2=(x0+bboxWidth, y0+bboxHeight), color=(255,255,255), thickness=1)   

        cv2.imshow('FirstImage', firstFrameOrigi) 
        cv2.waitKey(0)
        cv2.imshow('bluredImage', bluredImage) 
        cv2.waitKey(0)
        # cv2.imshow('diffImage', diffImage) 
        # cv2.waitKey(0)
        cv2.imshow('thresholdImage', thresholdImage) 
        cv2.waitKey(0)
        cv2.imshow('morphImage', morphImage) 
        cv2.waitKey(0)
        cv2.imshow('finalImage', firstFrame) 
        cv2.waitKey(0)


        
        # return firstFrame, p0

    #---------------------------------------------------------------------------------------------------------------------

    def detectObject(self, img):
        bluredImage = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=2)

        return bluredImage
    
    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)