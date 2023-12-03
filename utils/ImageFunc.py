import cv2
import os


basePath = "/home/yossi/Documents/database/hadar/videos"
relPath = "videoplayback.mp4"

vidPath = os.path.join(basePath,relPath)
print(f"{vidPath=}")

cv2.