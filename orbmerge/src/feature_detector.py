from __future__ import print_function
import numpy as np
from shutil import copyfile

import cv2
import  os

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.1


def alignImages(im1, im2, path):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(path + "matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    dirBagOne = '/home/user/ros/dip/logger/catkin_ws/bags/bag061827/'
    dirBagTwo = '/home/user/ros/dip/logger/catkin_ws/bags/bag062926/'
    os.makedirs('vers 0.2')

    for bagOneImage in os.listdir(dirBagOne):
        print("Reading reference image : ", bagOneImage)
        imReference = cv2.imread(dirBagOne + bagOneImage, cv2.IMREAD_COLOR)
        for bagTwoImage in os.listdir(dirBagTwo):
            print("Reading image to align : ", bagTwoImage)
            im = cv2.imread(dirBagTwo + bagTwoImage, cv2.IMREAD_COLOR)
            print("Aligning images ...")
            # Registered image will be resotred in imReg.
            # The estimated homography will be stored in h.
            try:
                os.makedirs('vers 0.2/' + bagOneImage + "-" + bagTwoImage)
                imReg, h = alignImages(im, imReference, 'vers 0.2/' + bagOneImage + "-" + bagTwoImage + '/')
                outFilename = 'vers 0.2/' + bagOneImage + "-" + bagTwoImage + '/' + bagOneImage + "-" + bagTwoImage + ".jpg"
                print("Saving aligned image : ", outFilename)
                copyfile(dirBagOne + bagOneImage, 'vers 0.2/' + bagOneImage + "-" + bagTwoImage + '/' + bagOneImage)
                copyfile(dirBagTwo + bagTwoImage, 'vers 0.2/' + bagOneImage + "-" + bagTwoImage + '/' + bagTwoImage)
                cv2.imwrite(outFilename, imReg)
                # Print estimated homography
                print("Estimated homography : \n", h)
            except cv2.error as e:
                print ("Error aligning " + bagOneImage + " with " + bagTwoImage)