# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import argparse
from nanosam.utils.predictor import Predictor
import cv2


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.engine")
    args = parser.parse_args()
        
    # Instantiate TensorRT predictor
    predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )

    # Read image and run image encoder
    image = PIL.Image.open("assets/frame2LargeHalf.png")
    # image = PIL.Image.open("assets/frame_target.jpg")

    if len(image.mode) == 4:
        image = image.convert('RGB') # rgb_image = rgba_image.convert('RGB')

    image_cv = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    # image_cv_downscaled = cv2.resize(image_cv, None, fx=0.5,fy=0.5)
    # image_cvHalf = image_cv[image_cv.shape[0]//2:,:]
    # cv2.imwrite("assets/frame2LargeHalf.png",image_cvHalf)

    Top_Left_X, Top_Left_Y, Width, Height = cv2.selectROI(image_cv) # [Top_Left_X, Top_Left_Y, Width, Height]
    bbox_cv = Top_Left_X, Top_Left_Y, Top_Left_X + Width, Top_Left_Y + Height

    # bbox_cv = list(np.array(bbox_cv) * 2)

    print(f"{bbox_cv=}")
       
    predictor.set_image(image)


    # Segment using bounding box
    bbox = [100, 100, 850, 759]  # x0, y0, x1, y1
    bbox = [100, 100, 1100, 759]  # x0, y0, x1, y1
    # bbox = [900, 800, 1050, 1050]  # x0, y0, x1, y1   
    # bbox = [0, 0, 1024, 1024]  # x0, y0, x1, y1
    bbox = bbox_cv
    
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[3]]
    ])   
    # points = np.array([
    #     [bbox[0] + Width//2, bbox[1] + Height//2]
    # ])

    point_labels = np.array([2,3])
    # point_labels = np.array([1])

    mask, _, _ = predictor.predict(points, point_labels)

    mask = (mask[0, 0] > 0).detach().cpu().numpy()

    # Draw resykts
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, 'g-')
    plt.savefig("data/frame2LargeHalf_roi.png")

