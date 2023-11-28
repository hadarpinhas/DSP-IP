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

        # relPath = r"videos\IR_Videos\IR_AIRPLANE_002.mp4"
        relPath = r"videos\IR_Videos\drone1_1m.mp4"

        outRelPath = r"videos\IR_Videos\outputVideos\output_drone1_1m.mp4"

        videoPath = os.path.join(basePath , relPath) 
        self.outputVideoPath = os.path.join(basePath , outRelPath)
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = -1 # in milliseconds, -1/0 for continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.frameCutFactor = 1 # to cut annotation like FLIR and other trademarks...
        
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

        if self.frameCutFactor != 1:
            frameWidth, frameHeight = int(firstFrame.shape[1] * self.frameCutFactor), int(firstFrame.shape[0] * self.frameCutFactor)
            firstFrame = firstFrame[firstFrame.shape[0] - self.frameHeight : self.frameHeight, firstFrame.shape[1] - self.frameWidth  : self.frameWidth ]
        else:
            frameWidth, frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(firstFrame.shape[1], firstFrame.shape[0]))

        while(True):
            _, newFrame = self.cap.read()
            t0 = time.time()
            if type(newFrame) == type(None):
                break
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            if self.frameCutFactor != 1:
                newFrame = newFrame[newFrame.shape[0] - frameHeight : frameHeight, newFrame.shape[1] - frameWidth  : frameWidth ]

            self.findBbox(newFrame)
            print(f"{(time.time() - t0) =}")
            print(f"{int((time.time() - t0) * 1000) =}")

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

        medianImage10 = cv2.blur(src=frameGray, ksize=(31,31))
        medianImage3  = cv2.blur(src=frameGray, ksize=(3,3))

        # diffImage = abs(frameGray - medianImage10)
        #diffImage = np.max(diffImage,0)

        ret, thresholdImage = cv2.threshold(medianImage3, 180, 255, cv2.THRESH_BINARY)
        # filterSize =(51, 51) 
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
        # tophat_img = cv2.morphologyEx(thresholdImage, cv2.MORPH_TOPHAT, kernel) 
        # print(f"{medianImage3.dtype=}")
        # print(f"{np.max(np.depth(medianImage3), cv2.CV_32F)=}")
        # LaplacePic = cv2.Laplacian(frameGray, ddepth=24, ksize=11) # int ktype = std::max(CV_32F, std::max(ddepth, src.depth()))

        boxSize = 9
        kernelSize = 31
        kernel = 0*np.ones((kernelSize,kernelSize),np.uint8)
        kernel[kernelSize//2-boxSize//2:kernelSize//2+boxSize//2+1, kernelSize//2-boxSize//2:kernelSize//2+boxSize//2+1] = 1
        # kernel = (kernel // np.sum(kernel))
        print(f"{kernel[:,kernel.shape[1]//2]}")
        print(f"{np.sum(kernel)=}")
        # top, bottom, left, right = borderSize, borderSize, borderSize, borderSize
        # kernel = cv2.copyMakeBorder(kernel, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,value=-1)
        # filter2DImage = cv2.filter2D(src=thresholdImage,ddepth=-1,kernel=kernel)
        corrMat = cv2.matchTemplate(image=medianImage3, templ=kernel, method=cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrMat)
        print(f"{max_loc=}")

        # diffImage = thresholdImage - filter2DImage
        # diffImage = np.where((diffImage>0),diffImage,0)

        # cv2.imshow("medianImage3", medianImage3)
        # cv2.waitKey(0)
        # cv2.imshow("kernel", kernel)
        # cv2.waitKey(0)          
        # cv2.imshow("corrMat", corrMat)
        # cv2.waitKey(0)
        pMax = max_loc[0] + kernelSize//2, max_loc[1] + kernelSize//2
        cv2.circle(frame, center=pMax, radius=2,color=(0,0,255), thickness=-1)
        # cv2.imshow("frame1", frame)
        # cv2.waitKey(0)


        # cv2.imshow("diffImage", diffImage)
        # cv2.waitKey(0)

        # numLabels, labels, stats, centroids =  cv2.connectedComponentsWithStats(image=thresholdImage, connectivity=4, ltype=cv2.CV_32S)
	    # # x,y,w,h,area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        # if numLabels > 1:
        #     areasList = stats[:, cv2.CC_STAT_AREA] 

        #     print(f"{stats=}")
        #     print(f"{areasList=}")
        #     maxAreaIdx = np.argmax(areasList[1:]) # index 0 is for background with the maximal area
        #     print(f"{maxAreaIdx=}")
        #     self.cX,self.cY = centroids[1:][maxAreaIdx]
        #     print(f"{self.cX,self.cY}")

        #     # finalImage = morphImage.copy()
    
        #     # (yi, xi) = np.unravel_index(finalImage.argmax(), finalImage.shape)

        #     # print(f"{xi=}, {yi=}")
        #     # x0, x1 = xi-self.bboxWidth,  xi+self.bboxWidth
        #     # y0, y1 = yi-self.bboxHeight, yi+self.bboxHeight

        #     # cv2.rectangle(img=frame, pt1=(x0, y0), pt2=(x1, y1), color=(255,0,0), thickness=1)   
        # cv2.circle(img=frame, center=(int(self.cX), int(self.cY)), radius=1, color=(0,0,255), thickness=-1)   

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)