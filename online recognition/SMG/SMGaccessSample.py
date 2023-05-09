#-------------------------------------------------------------------------------
# Name:        SMG dataset access sample
# Purpose:     Provide easy access to SMG data samples
#
# Author:      Haoyu Chen
#
# Created:     21/01/2019
# Copyright:   (c) Haoyu Chen 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import csv
import os
import shutil
import warnings
import zipfile

import cv2
import numpy


class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['SpineBase']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['SpineMid']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ShoulderCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['Head']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ShoulderLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ElbowLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['WristLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HandLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ShoulderRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ElbowRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['WristRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HandRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HipLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['KneeLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['AnkleLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['FootLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HipRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['KneeRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['AnkleRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['FootRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['SpineShoulder']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HandTipLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ThumbLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['HandTipRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))
        pos=pos+11
        self.joins['ThumbRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(float,data[pos+7:pos+9])))

    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel

class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName, sampleID, validation = 0):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.exists(fileName): #or not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        self.seqID = sampleID

        # Read skeleton data
        skeletonPath = os.path.join(fileName, sampleID + '_skeleton.csv')

        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]
        with open(skeletonPath, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.skeletons.append(Skeleton(row))
            del filereader
            # Read sample data
        sampleDataPath = os.path.join(fileName, sampleID + '_data.csv')

        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader


        labelsPath = os.path.join(fileName, sampleID + '_finelabels.csv')

        if not os.path.exists(labelsPath):
            warnings.warn("Labels are not available", Warning)
            self.labels=[]
        else:
            self.labels=[]
            f = open(labelsPath, "r")
            label_List = f.read().splitlines()
            # print(label_List)

            '''gather all the ges info in this sample'''
            # Iterate for each action in this sample
            for label_info in label_List:
                #print(label_info)
                # Get the gesture ID, and start and end frames for the gesture
                self.labels.append([int(label_info.split(',')[0])-1,int(label_info.split(',')[1]),int(label_info.split(',')[2])])
            # print(self.labels)

    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']

    def getMGFlag(self):
        """ Get the number of frames for this sample """
        MG_flag = numpy.zeros( self.data['numFrames'])
        for ges_info in self.labels:
            [gestureID, startFrame, endFrame] = ges_info
            # Get the gesture ID, and start and end frames for the gesture
            MG_flag[startFrame:endFrame] = 1
        return MG_flag

    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels

