import cv2
import numpy as np
import math as m

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture('./narrow_camera_teaser_9.avi')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
startx = 650
starty = 400
w = 900
h = 600
dx = 200
dy = 40

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('wiggle.mp4', fourcc, 30, (w,h))

# Setup the termination criteria,  
# either 15 iteration or move by 
# atleast 2 pt 
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2) 

top,bottom,left,right = 100, 100, 100, 100
frameNum, shiftX, shiftY = 0,0,0
maxShiftX, maxShiftY, counter = 0, 0, 0
while(cap.isOpened()):

  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame

    imageShift = frame

    if frameNum > 20:
      counter  +=1
      # if(counter%200 == 0):
      #     dx+=5
      if( (counter + 1)%2 == 0):
          dy-=200 
      if(  counter     %2  == 0):
          dy+=200 
      shiftX = int(                           5 * m.sin(counter/10) )
      shiftY = int( dy * m.exp(-counter/10) + 5 * m.sin(counter /10) )#  

    maxShiftX = max(maxShiftX, abs(shiftX))
    maxShiftY = max(maxShiftY, abs(shiftY))
    M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])

    # cut image for smaler ROI
    frameCut = frame[starty: starty+h,startx:startx + w,:]

    # Use warpAffine to perform the shift
    imageShift = cv2.warpAffine(frameCut, M, (w, h))

    strF1 = f"{frameNum=}, {counter=}, {shiftX=}, {shiftY=}" # {int(dy)=},  
    strF2 = f"{abs(maxShiftX)=}px, {abs(maxShiftY)=}px" # {int(dy)=},  
    cv2.putText(imageShift, strF1, (0,50),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
    cv2.putText(imageShift, strF2, (0,100),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
    cv2.putText(imageShift, 'scale_000px_', (0,150),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
    cv2.putText(imageShift, 'scale_100px_', (0,250),fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=2, color=(0,255,0))
    cv2.imshow('frameShifted',imageShift)
      
    out.write(imageShift)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xff == ord('q'):
      break


    frameNum +=1



  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()