import cv2
import numpy as np

def align_images_with_ecc(template: np.ndarray, inputImg: np.ndarray, centerRoiSize: int = 200, motion_type: int = cv2.MOTION_TRANSLATION) -> np.ndarray:

    """
    Aligns nextImg to prevImg using the Enhanced Correlation Coefficient (ECC) algorithm.

    Parameters:
    - prevImg: The reference image to which nextImg will be aligned.
    - nextImg: The image that will be aligned to prevImg.
    - motion_type: The type of motion to be used. Default is cv2.MOTION_EUCLIDEAN.
                   Other options include cv2.MOTION_TRANSLATION, cv2.MOTION_AFFINE, etc.
                   cv2.MOTION_TRANSLATION   - > 2D shift x,y
                   cv2.MOTION_EUCLIDEAN     - > 2D shift x,y , rotation
                   cv2.MOTION_AFFINE        - > 2D shift x,y , rotation, scaling
                   cv2.MOTION_HOMOGRAPHY    - > 3D

    Returns:
    - aligned_image: The second image aligned to the first image.
    """
    image1 = template.copy()
    image2 = inputImg.copy()

    # Convert images to grayscale for the ECC algorithm
    if image1.ndim == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
    if image2.ndim == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2

    centerX, centerY =  gray1.shape[1] // 2, gray1.shape[0] // 2

    gray1 = cv2.getRectSubPix(image=gray1, patchSize=(centerRoiSize, centerRoiSize), center=(centerX, centerY))
    gray2 = cv2.getRectSubPix(image=gray2, patchSize=(centerRoiSize, centerRoiSize), center=(centerX, centerY))

    cv2.imshow('gray2', gray2)

    # Depending on the motion type, initialize the warp matrix accordingly
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define criteria to terminate the algorithm (specify the maximum number of iterations and/or the desired accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001)

    # Apply the ECC algorithm to find the best transformation matrix
    try:
        # Find the transformation matrix that aligns nextImg to prevImg
        cc, warp_matrix = cv2.findTransformECC(templateImage=gray1, inputImage=gray2,warpMatrix=warp_matrix,motionType=motion_type,criteria=criteria)
        print(warp_matrix)

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            aligned_image = cv2.warpPerspective(inputImg, warp_matrix, (inputImg.shape[1], inputImg.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean/Rigid, and Affine
            aligned_image = cv2.warpAffine(inputImg, warp_matrix, (inputImg.shape[1], inputImg.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)


    except cv2.error as e:
        print(f"Error: {e}")
        aligned_image = None

    # cv2.imshow('gray1', gray1)
    # aligned_image2 = cv2.warpAffine(gray2, warp_matrix, (gray2.shape[1], gray2.shape[0]), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    # cv2.imshow('aligned_image2', aligned_image2)
    # cv2.waitKey(0)

    return aligned_image

    #------------------------------------------------------------------------------------------------------

def match(baseImg: np.ndarray, templateImg: np.ndarray, centerRoiSize: int = 400) -> tuple[np.float32, tuple[np.uint8, np.uint8]]:

    gradTemplate = doSobel(templateImg)
    gradBase     = doSobel(baseImg)

    centerX, centerY =  gradBase.shape[1] // 2, gradTemplate.shape[0] // 2

    margin = centerRoiSize//10

    gradBase        = cv2.getRectSubPix(image=gradBase,     patchSize=(centerRoiSize + margin, centerRoiSize + margin), center=(centerX, centerY))
    gradTemplate    = cv2.getRectSubPix(image=gradTemplate, patchSize=(centerRoiSize                    , centerRoiSize                    ), center=(centerX, centerY))

    # res = Correlation matrix
    res = cv2.matchTemplate(gradBase, gradTemplate, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    max_loc = [max_loc[0]-margin//2, max_loc[1]-margin//2]

    gradBase = gradBase[margin//2:gradBase.shape[0]-margin//2, margin//2:gradBase.shape[1]-margin//2]
    cv2.imshow('gradBase', gradBase)
    translation_matrix = np.float32([ [1,0,max_loc[0]], [0,1,max_loc[1]] ])
    t = 0                    # padSizeTop
    b = margin    # padSizeBottom
    l = 0                    # padSizeLeft
    r = margin  # padSizeRight
    aligned_image2 = cv2.warpAffine(gradTemplate, translation_matrix, gradTemplate.shape[1::-1])
    cv2.imshow('aligned_image2', aligned_image2)

    res = res.astype(float) * 255

    print(f"{aligned_image2.shape=}")
    print(f"{gradBase.shape=}")
    delta = cv2.absdiff(gradBase, aligned_image2)
    cv2.imshow('delta', delta*10)

    # paddedCorrMat = cv2.copyMakeBorder(res, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return max_val, max_loc # , gradTemplate, gradBase

    #------------------------------------------------------------------------------------------------------

def doSobel(src):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.GaussianBlur(src, (3, 3), 0)

    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x) # =abs(scaledImg).astype(np.uint8). default: scales with alpah=1 beta=0 scaledImg=Img*alpha+beta. 
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    return grad