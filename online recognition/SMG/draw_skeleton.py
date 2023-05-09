
#-------------------------------------------------------------------------------
# Name:        THD-HMM for gesture recognition utilities
# Purpose:     provide some toolkits
# Copyright (C) 2018 Haoyu Chen <chenhaoyucc@icloud.com>,
# author: Chen Haoyu

# center of Machine Vision and Signal Analysis,
# Department of Computer Science and Engineering,
# University of Oulu, Oulu, 90570, Finland

# this code is based on the Starting Kit for ChaLearn LAP 2014 Track3
# and Di Wu: stevenwudi@gmail.com DBN implement for CVPR
# thanks for their opensource
#-------------------------------------------------------------------------------
""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

"""
#coding=utf-8
from time import sleep
import shutil
import gc
import copy
import numpy
import random
import cv2
from PIL import Image, ImageDraw
import os
import time
from iMiGUEaccessSample import GestureSample

'''
load iMiGUE training dataset
'''
parsers = {
'data': "./mg_skeleton_only/",
#/media/micro-gesture/data/dataset/iMiGUE_skeleton/mg_skeleton_only-001
#used joint list
#'used_joints': ['HipCenter', 'ShoulderCenter', 'Head',
#   'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
#   'ShoulderRight','ElbowRight','WristRight','HandRight'],
'used_joints':
['Nose','ShoulderCenter','ShoulderLeft','ElbowLeft','HandLeft','ShoulderRight','ElbowLeft','HandRight','EyeLeft',
'EyeRight','EarLeft','EarRight']}
# start counting tim
time_tic = time.time()
#  348 347 319 218
#how many joints are used
njoints = len(parsers['used_joints'])

#print(njoints)

used_joints = parsers['used_joints']
training_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 73, 75, 76, 82, 83, 84, 86, 87, 89, 91, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 107, 108, 113, 115, 117, 119, 120, 121, 122, 124, 131, 132, 134, 136, 137, 140, 141, 143, 145, 147, 149, 151, 152, 153, 157, 158, 159, 160, 162, 164, 165, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 191, 193, 195, 197, 199, 214, 216, 222, 223, 225, 230, 231, 232, 233, 234, 236, 239, 240, 246, 247, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 260, 261, 263, 264, 266, 267, 268, 269, 270, 272, 273, 275, 276, 277, 279, 282, 283, 289, 290, 291, 293, 294, 296, 297, 298, 299, 300, 301, 302, 304, 305, 306, 308, 310, 311, 312, 313, 314, 315, 316, 317, 323, 324, 325, 328, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 433,420, 419, 407 , 350, 357, 358, 359, 360, 363, 365, 366, 367, 369, 374, 376, 377, 379, 381, 382, 386, 392, 394, 397, 399, 400, 401, 402, 403, 404, 405, 407, 408, 430, 435]

testing_list = [54, 67, 69, 70, 71, 72, 74, 77, 79, 85, 98, 105, 106, 109, 110, 111, 112, 114, 116, 118, 125, 126, 133, 135, 138, 144, 148, 154, 156, 161, 163, 166, 167, 168, 169, 196, 198, 200, 201, 205, 208, 212, 213, 220, 235, 237, 238, 241, 242, 243, 244, 262, 274, 280, 292, 303, 318, 320, 322, 327, 329, 346, 349, 355, 368, 371, 375, 378, 380, 385, 396, 406, 410, 411, 413, 414, 415, 416, 417, 421, 423, 424, 425, 428, 429, 431, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453]

washed_testing_list =  [72, 79, 106, 109, 110, 111, 112, 114, 118, 126, 133, 148, 154, 161, 166, 168, 169, 196, 198, 200, 201, 205, 208, 212, 213, 220, 235, 237, 238, 241, 242, 243, 244, 274, 280, 292, 303, 320, 322, 327, 346, 355, 368, 371, 375, 378, 380, 385, 406, 411, 414, 415, 417, 421, 423, 424, 429, 436, 439, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453]

high_washed_testing_list =  [72, 79, 118, 133, 166, 196, 198, 200, 201, 205, 213, 238, 243, 274, 280, 292, 320, 327, 346, 355, 368, 371, 375, 380, 385, 406, 414, 415, 423, 424, 429, 439, 441, 442, 443, 444, 445, 446, 448, 449, 450, 451, 452, 453]

# training_list = [1, 2, 3, 4]
training_list = sorted(training_list, reverse=True)
#start traversing samples

# gesture_list = ['缩脖子','鼓起脸','摸帽子','摸脸','摸前额','盖住脸',
# '揉眼睛','摸脸部分','摸耳朵','咬指甲','摸下巴','摸脖子',
# '玩头发','领带','摸胸口','抓后背','折手臂','鲁袖子',
# '手臂放后面','移动躯干','坐直','抓手臂','撮手','交叉手指',
# '尖塔式手势','玩东西','撤回手臂','抬头','咬嘴唇','叉腰',
# '耸肩','illustrative']

gesture_list = ['turtle neck','deep breath','touch hat','touch face','touch forehead','cover face',
'rub eyes','touch other face part','touch ears','bite nails','touch chin','touch neck',
'play hair','tie','chest','back','fold arm','sleeve',
'put arm back','move torso','sit straight','sractch arm','rub hand','cross finger',
'mineat hand','play things','withdraw arms','head up','bit lip','arm akimbo',
'shake shoulder','illustrative']



fps = 60
for sampleID in testing_list:

    print("\t Processing file " + str(sampleID))

    sampleID = str(sampleID).zfill(4)
    smp=GestureSample(os.path.join(parsers['data'],sampleID),sampleID)

    gesturesList=smp.getGestures()

    '''traverse all ges in samples'''
    ges_index = 0
    #get the skeletons of the gesture
    de_flag = 0
    for ges_info in gesturesList:
        gestureID, startFrame, endFrame = ges_info

        ges_index += 1
        # print(endFrame - startFrame)
        repeat_flag = 1


        while repeat_flag == 1:
            for numFrame in range(startFrame+1,endFrame+1):
                skel=smp.getSkeleton(numFrame)

                if int(sum(skel.joins['HandRight']))== 0 or int(sum(skel.joins['HandLeft'])) ==0:
                    #hand_null ==True
                    print('bad hand frame')
                    repeat_flag = 0
                    #break

                # Get the Skeleton object for this frame
                skelImage=smp.getSkeletonImage(numFrame)
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,400)
                fontScale              = 1
                fontColor              = (128,128,128)
                thickness              = 3
                lineType               = 1

                cv2.putText(skelImage,gesture_list[gestureID],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

                winname = "Test"
                #cv2.namedWindow(winname)        # Create a named window
                #cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
                cv2.imshow(sampleID + str(ges_index),skelImage)
                sleep(1/fps)
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            user_input = input("Enter r=repeat\n")
            if user_input == "r":
                print("Repeating video")
                repeat_flag = 1
            elif user_input == "d":
                print("decided sample video")
                de_flag = 1
            else:
                repeat_flag = 0
        if de_flag == 1:
            break
