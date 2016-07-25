import cv2
import numpy as np
import os, sys

def detectAndDescribe(image):
    # convert the image to grayscale
    gray = image

    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    img1 = cv2.drawKeypoints(image, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("img1.jpg", img1)
    # convert the keypoints from KeyPoint objects to NumPy arrays
    nkps = np.float32([kp.pt for kp in kps])
    
    # return a tuple of keypoints and features
    return (kps, nkps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    # compute the raw matches and initialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None

def match(imageA, imageB, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
    
    (kps1,kpsA, featuresA) = detectAndDescribe(imageA)
    (kps2,kpsB, featuresB) = detectAndDescribe(imageB)
    # match features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    (matches, H, status) = M
    matches=sorted(matches, key= lambda x:x.distance)
    if M is None:
        return None

    return kps1, kps2, matches[:21] 

'''
if __name__=="__main__":
    imageA = cv2.imread("./images/images/chinese_view.jpg", 0)
    imageB = cv2.imread("./images/images/source/flower.jpg", 0)
    matched_img1 = match(imageA, imageB)
    cv2.imwrite("matched.jpg", matched)
'''
