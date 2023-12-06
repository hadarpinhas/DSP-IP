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
               
        basePath = r"/home/yossi/Documents/database/hadar" # for yossi pc
        # basePath = r"C:\Users\User\Documents\dataBase\DSP-IP" # for Hadar's personal pc

        # relPath = r"videos\IR_Videos\IR_AIRPLANE_002.mp4"
        relPath = r"videos/IR_Videos/drone1_1m.mp4"
        relPath = r"videos/IR_Videos/saved_drone1_1m_grayLevel150.mp4"

        outRelPath = r"videos/outputVideos/output_drone1_1m.mp4"

        videoPath = os.path.join(basePath , relPath) 
        self.outputVideoPath = os.path.join(basePath , outRelPath)
        print(f"{videoPath=}")
        print(f"{self.outputVideoPath=}")

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.outputFps = 30.0

        self.cap = cv2.VideoCapture(videoPath)
        self.startDetectionFrame = 0 # 

        self.waitKeyParameter = 1 # in milliseconds, -1/0 for paused, alternatively, 1 milisecond continuous

        self.to_draw_line = False        

        self.screenSizeFactor = 1 # 0.9 for my laptop smaller screen

        self.frameCutFactor = 1 # to cut annotations built-in on images like FLIR and other trademarks...

        self.imageDownScaleFactor = 0.05 # for detection in images with large pixels size (low resolution)

    #---------------------------------------------------------------------------------------------------------------------

    def runTracker(self):

        self.initVideoParams()

        while(True):
            _, newFrame = self.cap.read()

            if type(newFrame) == type(None):
                break
            if self.screenSizeFactor != 1: # for my laptop smaller screen
                newFrame = cv2.resize(newFrame, None, fx=self.screenSizeFactor,fy=self.screenSizeFactor)

            if self.frameCutFactor != 1:
                newFrame = newFrame[newFrame.shape[0] - frameHeight : frameHeight, newFrame.shape[1] - frameWidth  : frameWidth ]

            frameGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

            pMax = self.findBbox(frameGray)

            annotatedFrame = cv2.circle(newFrame, center=pMax, radius=max(2, 2*self.imageDownScaleFactor),color=(0,0,255), thickness=1)

            # annotatedFrame = cv2.circle(newFrame, center=pMax, radius=max(12, 2*self.imageDownScaleFactor),color=(5,5,5), thickness=-1)
            # dgl = 150 # drone Gray Level
            # annotatedFrame = cv2.circle(newFrame, center=pMax, radius=max(1, 2*self.imageDownScaleFactor),color=(dgl,dgl,dgl), thickness=-1)

            cv2.imshow('frame', annotatedFrame) 

            # self.outputVideo.write(newFrame)
               
            k = cv2.waitKey(self.waitKeyParameter)
            if k == 27: 
                break

        cv2.destroyAllWindows() 
        self.cap.release()
        self.outputVideo.release()

    #---------------------------------------------------------------------------------------------------------------------
       
    def initVideoParams(self):

        frameNumber = 0
        success, firstFrame = self.cap.read()
        while(frameNumber < self.startDetectionFrame) and success: # skip irrelevant frames
            _, firstFrame = self.cap.read()            
            frameNumber += 1

        if not success:
            sys.exit("No first Frame! chack data/video (data source path) please.")

        if self.frameCutFactor != 1:
            frameWidth, frameHeight = int(firstFrame.shape[1] * self.frameCutFactor), int(firstFrame.shape[0] * self.frameCutFactor)
            firstFrame = firstFrame[firstFrame.shape[0] - self.frameHeight : self.frameHeight, firstFrame.shape[1] - self.frameWidth  : self.frameWidth ]
        else:
            frameWidth, frameHeight = firstFrame.shape[1], firstFrame.shape[0]

        self.outputVideo = cv2.VideoWriter(filename=self.outputVideoPath, fourcc=self.fourcc, fps=self.outputFps, frameSize=(firstFrame.shape[1], firstFrame.shape[0]))

    #---------------------------------------------------------------------------------------------------------------------
        
    def findBbox(self, frameGray):
        median3 = cv2.blur(frameGray, ksize=(3,3))
        frameGrayDs = cv2.resize(median3, None, fx=self.imageDownScaleFactor, fy=self.imageDownScaleFactor, interpolation=cv2.INTER_AREA)
        ret, thresholdImage = cv2.threshold(frameGrayDs, 50, 255, cv2.THRESH_BINARY)
        
        cv2.imshow("frameGrayDs", cv2.resize(frameGrayDs, None, fx=1/self.imageDownScaleFactor, fy=1/self.imageDownScaleFactor, interpolation=cv2.INTER_AREA))
        cv2.waitKey(0)
        cv2.imshow("thresholdImage", thresholdImage)
        cv2.waitKey(0)
        # boxSide = int((9 * self.imageDownScaleFactor)//2)

        boxSide = 1
        # kernelSize = max(9, int(np.ceil(31 * self.imageDownScaleFactor)))
        kernelSize = 9
        kernel = 0*np.ones((kernelSize,kernelSize),np.uint8)
        kernel[kernelSize//2-boxSide+1:kernelSize//2+boxSide, kernelSize//2-boxSide+1:kernelSize//2+boxSide] = 1

        print(f"{kernel[:,kernel.shape[1]//2]=}")
        print(f"{np.sum(kernel)=}")

        corrMat = cv2.matchTemplate(image=thresholdImage, templ=kernel, method=cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrMat)
        print(f"{max_loc=}")

        pMax = max_loc[0] + kernelSize//2, max_loc[1] + kernelSize//2

        pMax = self.getFineTune(frameGray, pMax, (frameGrayDs.shape[0], frameGrayDs.shape[1]))

        return pMax

    #---------------------------------------------------------------------------------------------------------------------
    
    def getFineTune(self, frameGray, pMaxDs, originShape):
            
        fineTuneMatchBox = 100

        pMaxUpscaled = (int(pMaxDs[0] * frameGray.shape[1]/originShape[1]), int(pMaxDs[1] * frameGray.shape[0]/originShape[0]))
        print(f"{pMaxDs=}")
        print(f"{pMaxUpscaled=}")

        fineTuneSmallFrame = frameGray[ pMaxUpscaled[1]-fineTuneMatchBox//2:pMaxUpscaled[1]+fineTuneMatchBox//2,
                                        pMaxUpscaled[0]-fineTuneMatchBox//2:pMaxUpscaled[0]+fineTuneMatchBox//2 ]

        ret,thresh = cv2.threshold(fineTuneSmallFrame,150,255,0)     
        # calculate moments of binary image
        M = cv2.moments(thresh)     
        # calculate x,y coordinate of center mji=∑x,y(array(x,y)⋅xj⋅yi), xj=x^j. yi=y^i
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        pMax = pMaxUpscaled[0] + cX - fineTuneMatchBox//2, pMaxUpscaled[1] + cY - fineTuneMatchBox//2

        return pMax

    #---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ObjectDetection)
    unittest.TextTestRunner(verbosity=0).run(suite)








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