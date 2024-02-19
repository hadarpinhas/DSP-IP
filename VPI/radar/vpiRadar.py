import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import time
import cv2

class VpiRadar:

    # ----------------------------
    # Parse command line arguments
    # parser = ArgumentParser()
    # parser.add_argument('backend', choices=['cpu','cuda'],
    #                     help='Backend to be used for processing')

    # # parser.add_argument('input',
    # #                     help='Input image in space domain')

    # args = parser.parse_args();

    # if args.backend == 'cpu':
    #     backend = vpi.Backend.CPU
    # else:
    #     assert args.backend == 'cuda'
    #     backend = vpi.Backend.CUDA


    def calcFft(self, dataPath, backend=vpi.Backend.CPU):

        t0 = time.time()

        cap = cv2.VideoCapture(dataPath)
        while(True):
            _, newFrame = cap.read()
            if type(newFrame) == type(None):
                break

            # --------------------------------------------------------------
            # Load input into a vpi.Image and convert it to float grayscale
            with vpi.Backend.CUDA:
                try:
                    input = vpi.asimage(newFrame).convert(vpi.Format.F32)
                except IOError:
                    sys.exit("Input file not found")
                except:
                    sys.exit("Error with input file")

            # --------------------------------------------------------------
            # Transform input into frequency domain
            with backend:
                hfreq = input.fft()


        cv2.destroyAllWindows() 
        cap.release()
        # self.outputVideo.release()

        tTotal = (time.time() - t0) *1e3
        print(f"The time took for fft with {repr(backend)} is {tTotal:.1f} ms")


if __name__ == "__main__":

    vpiRadar = VpiRadar()

    dataPath = r'/home/yossi/Documents/database/hadar/vpiVideo/dashcam.mp4'

    vpiRadar.calcFft(dataPath=dataPath,backend=vpi.Backend.CPU)

    vpiRadar.calcFft(dataPath=dataPath, backend=vpi.Backend.CUDA)


# # --------------------------------------------------------------
# # Post-process results and save to disk

# # Transform [H,W,2] float array into [H,W] complex array
# hfreq = hfreq.cpu().view(dtype=np.complex64).squeeze(2)

# # Complete array into a full hermitian matrix
# if input.width%2==0:
#     wpad = input.width//2-1
#     padmode = 'reflect'
# else:
#     wpad = input.width//2
#     padmode='symmetric'
# freq = np.pad(hfreq, ((0,0),(0,wpad)), mode=padmode)
# freq[:,hfreq.shape[1]:] = np.conj(freq[:,hfreq.shape[1]:])
# freq[1:,hfreq.shape[1]:] = freq[1:,hfreq.shape[1]:][::-1]

# # Shift 0Hz to image center
# freq = np.fft.fftshift(freq)

# # Convert complex frequencies into log-magnitude
# lmag = np.log(1+np.absolute(freq))

# # Normalize into [0,255] range
# min = lmag.min()
# max = lmag.max()
# lmag = ((lmag-min)*255/(max-min)).round().astype(np.uint8)

# # -------------------
# # Save result to disk
# Image.fromarray(lmag).save('spectrum_python'+str(sys.version_info[0])+'_'+args.backend+'.png')
