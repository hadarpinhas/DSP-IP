
from pathlib import Path
import cv2
from skimage.measure import block_reduce
import numpy as np
import time
import unittest

class DogFilter(unittest.TestCase):
    
    def setUp(self):
        self.startTime = time.time()
        self.startTimeHH = time.strftime('%H:%M:%S')

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.5f minutes. Started at:%s' % (self.id(), (t/60), (self.startTimeHH)))

    def testOne(self):
        print("-"*50 + '\n' + "running DogFilter main" + '\n' + "-"*50)

        self.workPath = Path(r"C:\Users\User\Documents\dataBase\featuresResults")
        
        self.main()

    #--------------------------------------------------------------------
  
    def main(self):

        usafPath = self.workPath / "ZKfAGS.jpeg"
        
        usafImg  = cv2.imread(str(usafPath), cv2.IMREAD_GRAYSCALE)

        # here we downscale the orignal image to later upscale, to simulate an image with smaller pixel size.
        # down scale orginal image
        usafImgDs = cv2.resize(usafImg, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA    )
        # Up scale the DS
        usafImgUs = cv2.resize(usafImgDs,   None, fx=4  ,   fy=4  , interpolation=cv2.INTER_NEAREST )

        otsu_threshold, OtsuFilteredImage = cv2.threshold(usafImgUs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #the second paprameter means nothing, is overrun by otsu


        # blockSize	Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        # C	Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        adaptMeanFilteredImage = cv2.adaptiveThreshold(usafImgUs,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,blockSize=7, C=2)
        adaptGausFilteredImage = cv2.adaptiveThreshold(usafImgUs,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=7 ,C=2)
        DogFilteredImage   = self.XDoG_filter(usafImgUs)
        CannyFilteredImage = cv2.Canny(adaptMeanFilteredImage,150,250, apertureSize = 5, L2gradient = False) # InputImage,threshold1,threshold2,apertureSize = 3,L2gradient = false

        OpenDogFilteredImage =  cv2.morphologyEx(DogFilteredImage, cv2.MORPH_OPEN, kernel=np.ones((3,3),np.uint8)) # erosion followed by dilation


        cv2.imwrite(str(self.workPath / "usafImgDsArea.png"     ), usafImgDs            )
        cv2.imwrite(str(self.workPath / "usafImgUsNearest.png"  ), usafImgUs            )
        cv2.imwrite(str(self.workPath / "OtsuFilteredImage.png" ), OtsuFilteredImage    )
        cv2.imwrite(str(self.workPath / "adaptMeanFilteredImage.png" ), adaptMeanFilteredImage    )
        cv2.imwrite(str(self.workPath / "adaptGausFilteredImage.png" ), adaptGausFilteredImage    )
        cv2.imwrite(str(self.workPath / "DogFilteredImage.png"  ), DogFilteredImage     )
        cv2.imwrite(str(self.workPath / "CannyFilteredImage.png"), CannyFilteredImage   )
        
        cv2.imwrite(str(self.workPath / "OpenDogFilteredImage.png"), OpenDogFilteredImage   )

    #--------------------------------------------------------------------
# https://github.com/Kazuhito00/XDoG-OpenCV-Sample/blob/main/XDoG.py
    def XDoG_filter(self, image, kernel_size=0,sigma=1.4, k_sigma=1.6, epsilon=0, phi=1, gamma=0.98):
        """XDoG(Extended Difference of Gaussians)

        Args:
            image: OpenCV Image
            kernel_size: Gaussian Blur Kernel Size
            sigma: sigma for small Gaussian filter
            k_sigma: large/small for sigma Gaussian filter
            eps: threshold value between dark and bright
            phi: soft threshold
            gamma: scale parameter for DoG signal to make sharp

        Returns:
            Image after applying the XDoG.
        """
        epsilon /= 255
        dog = self.DoG_filter(image, kernel_size, sigma, k_sigma, gamma)
        dog /= dog.max()
        e = 1 + np.tanh(phi * (dog - epsilon))
        e[e >= 1] = 1
        return e.astype('uint8') * 255

    #--------------------------------------------------------------------

    def DoG_filter(self, image, kernel_size=0, sigma=1.4, k_sigma=1.1, gamma=1):
        """DoG(Difference of Gaussians)

        Args:
            image: OpenCV Image
            kernel_size: Gaussian Blur Kernel Size
            sigma: sigma for small Gaussian filter
            k_sigma: large/small for sigma Gaussian filter
            gamma: scale parameter for DoG signal to make sharp

        Returns:
            Image after applying the DoG.
        """
        g1 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        g2 = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma * k_sigma)
        return g1 - gamma * g2

    #--------------------------------------------------------------------

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(DogFilter)
    unittest.TextTestRunner(verbosity=0).run(suite)