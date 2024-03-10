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
w = 600
h = 600
dx = 200
dy = 40

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('wiggle.mp4', fourcc, 30, (w,h))

frameNum=0
while(cap.isOpened()):
  frameNum +=1
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame

    if(frameNum%200 == 0):
        dx+=5
    if(frameNum%50 == 0):
        dy-=2
    tdx = int(dx * m.sin(frameNum/10))
    tdy = int(dy * m.cos(frameNum/10))

    f = frame[starty + tdy:starty+h + tdy,startx + tdx:startx + w + tdx,:]
    cv2.imshow('Frame',f)
    out.write(f)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xff == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
