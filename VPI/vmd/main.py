# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import vpi
import numpy as np
from os import path
from argparse import ArgumentParser
from contextlib import contextmanager
import cv2


# --------------------------------------
# Some definitions and utility functions

# Maximum number of keypoints that will be tracked
MAX_KEYPOINTS = 100

def update_mask(mask, trackColors, prevFeatures, curFeatures, status = None):
    '''Draw keypoint path from previous frame to current one'''

    numTrackedKeypoints = 0

    def none_context(a=None): return contextmanager(lambda: (x for x in [a]))()

    with curFeatures.rlock_cpu(), \
         (status.rlock_cpu() if status else none_context()), \
         (prevFeatures.rlock_cpu() if prevFeatures else none_context()):

        for i in range(curFeatures.size):
            # keypoint is being tracked?
            if not status or status.cpu()[i] == 0:
                color = tuple(trackColors[i,0].tolist())

                # OpenCV 4.5+ wants integers in the tuple arguments below
                cf = tuple(np.round(curFeatures.cpu()[i]).astype(int))

                # draw the tracks
                if prevFeatures:
                    pf = tuple(np.round(prevFeatures.cpu()[i]).astype(int))
                    cv2.line(mask, pf, cf, color, 2)

                cv2.circle(mask, cf, 5, color, -1)

                numTrackedKeypoints += 1

    return numTrackedKeypoints

def save_file_to_disk(frame, mask, baseFileName, frameCounter):
    '''Apply mask on frame and save it to disk'''

    frame = frame.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)
    with frame.rlock_cpu() as frameData:
        frame = cv2.add(frameData, mask)

    name, ext = path.splitext(baseFileName)
    fname = "{}_{:04d}{}".format(name, frameCounter, ext)

    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

# ----------------------------
# Parse command line arguments

parser = ArgumentParser()
parser.add_argument('backend', choices=['cpu', 'cuda'],
                    help='Backend to be used for processing')

parser.add_argument('input',
                    help='Input video to be processed')

parser.add_argument('pyramid_levels', type=int,
                    help='Number of levels in the pyramid used with the algorithm')

parser.add_argument('output',
                    help='Output file name')

args = parser.parse_args();

if args.backend == 'cuda':
    backend = vpi.Backend.CUDA
else:
    assert args.backend == 'cpu'
    backend = vpi.Backend.CPU

# adjust output file name to take into account backend used and python version
name, ext = path.splitext(args.output)
args.output = "{}_python{}_{}{}".format(name, sys.version_info[0], args.backend, ext)

# ----------------
# Open input video

inVideo = cv2.VideoCapture(args.input)

# Read first input frame
ok, cvFrame = inVideo.read()
if not ok:
    exit('Cannot read first input frame')

# ---------------------------
# Perform some pre-processing

# Retrieve features to be tracked from first frame using
# Harris Corners Detector
with vpi.Backend.CPU:
    frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8)
    curFeatures, scores = frame.harriscorners(strength=0.1, sensitivity=0.01)

# Limit the number of features we'll track and calculate their colors on the
# output image
with curFeatures.lock_cpu() as featData, scores.rlock_cpu() as scoresData:
    # Sort features in descending scores order and keep the first MAX_KEYPOINTS
    ind = np.argsort(scoresData, kind='mergesort')[::-1]
    featData[:] = np.take(featData, ind, axis=0)
    curFeatures.size = min(curFeatures.size, MAX_KEYPOINTS)

    # Keypoints' have different hues, calculated from their position in the first frame
    trackColors = np.array([[(int(p[0]) ^ int(p[1])) % 180,255,255] for p in featData], np.uint8).reshape(-1,1,3)
    # Convert colors from HSV to RGB
    trackColors = cv2.cvtColor(trackColors, cv2.COLOR_HSV2BGR).astype(int)

with backend:
    optflow = vpi.OpticalFlowPyrLK(frame, curFeatures, args.pyramid_levels)

# Counter for the frames
idFrame = 0

# Create mask with features' tracks over time
mask = np.zeros((frame.height, frame.width, 3), np.uint8)
numTrackedKeypoints = update_mask(mask, trackColors, None, curFeatures)

while True:
    # Apply mask to frame and save it to disk
    save_file_to_disk(frame, mask, args.output, idFrame)

    print("Frame id={}: {} points tracked.".format(idFrame, numTrackedKeypoints))

    prevFeatures = curFeatures

    # Read one input frame
    ret, cvFrame = inVideo.read()
    if not ret:
        print("Video ended.")
        break
    idFrame += 1

    # Convert frame to grayscale
    with vpi.Backend.CUDA:
        frame = vpi.asimage(cvFrame, vpi.Format.BGR8).convert(vpi.Format.U8);

    # Calculate where keypoints are in current frame
    curFeatures, status = optflow(frame)

    # Update the mask with the current keypoints' position
    numTrackedKeypoints = update_mask(mask, trackColors, prevFeatures, curFeatures, status)

    # No more keypoints to track?
    if numTrackedKeypoints == 0:
        print("No keypoints to track.")
        break # nothing else to do
