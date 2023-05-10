#-------------------------------------------------------------------------------
# Name:        MiGA evaluation scripts
# Purpose:     Provide evaluation scripts for MiGA challenge tracks
#
# Author:      Haoyu Chen
#
# Created:     21/03/2023
# Copyright:   (c) MiGA IJCAI 2023
# Licence:     GPL
#-------------------------------------------------------------------------------
import csv
import os
import shutil
import warnings
import zipfile
import openpyxl
import cv2
import numpy
import pandas as pd
import xlrd
from PIL import Image, ImageDraw

class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        self.data_norm  = max(data) - min(data)
        pos=1
        self.joins['Nose']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['ShoulderCenter']=(list(map(float,data[pos:pos+3])))
        #2
        pos=pos+3
        self.joins['ShoulderLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['ElbowLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['HandLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['ShoulderRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['ElbowRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['HandRight']=(list(map(float,data[pos:pos+3])))

        #8
        pos=pos+3
        self.joins['HipCenter']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['HipLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['KneeLeft']=(list(map(float,data[pos:pos+3])))
        #11
        pos=pos+3
        self.joins['AnkleLeft']=(list(map(float,data[pos:pos+3])))

        #12
        pos=pos+3
        self.joins['HipRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['KneeRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['AnkleRight']=(list(map(float,data[pos:pos+3])))

        #15
        pos=pos+3
        self.joins['EyeLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['EyeRight']=(list(map(float,data[pos:pos+3])))

        #17
        pos=pos+3
        self.joins['EarLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['EarRight']=(list(map(float,data[pos:pos+3])))

        #19
        pos=pos+3
        self.joins['ToeLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['SmallToeLeft']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['HeelLeft']=(list(map(float,data[pos:pos+3])))

        pos=pos+3
        self.joins['ToeRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['SmallToeRight']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['HeelRight']=(list(map(float,data[pos:pos+3])))


        # right hands
        pos=pos+3
        self.joins['RightFinger1']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['RightFinger2']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['RightFinger3']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['RightFinger4']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['RightFinger5']=(list(map(float,data[pos:pos+3])))



        # left hands
        pos=pos+3
        self.joins['LeftFinger1']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['LeftFinger2']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['LeftFinger3']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['LeftFinger4']=(list(map(float,data[pos:pos+3])))
        pos=pos+3
        self.joins['LeftFinger5']=(list(map(float,data[pos:pos+3])))




    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            # print(self.joins[key])
            skel[key]=self.joins[key]
            # print(self.joins[key][0])

        return skel
    def getHandnull(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            # print(self.joins[key])
            skel[key]=self.joins[key]
            # print(self.joins[key][0])

        return skel, hand_null
        if int(sum(self.joins['HandRight'] +self.joins['HandLeft']))== 0:
            hand_null ==True

    def getJoiientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key]
        return skel

    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['ShoulderCenter','Nose'],['ShoulderCenter','ShoulderLeft'],['ShoulderLeft','ElbowLeft'],['ElbowLeft','HandLeft'],
        ['ShoulderCenter','ShoulderRight'],['ShoulderRight','ElbowRight'],['ElbowRight','HandRight'], ['Nose','EyeLeft'],['EyeLeft','EarLeft'],
        ['Nose','EyeRight'],['EyeRight','EarRight'],['HandLeft','LeftFinger1'],['HandLeft','LeftFinger2'],['HandLeft','LeftFinger3'],['HandLeft','LeftFinger4'],
        ['HandLeft','LeftFinger5'],['HandRight','RightFinger1'],['HandRight','RightFinger2'],['HandRight','RightFinger3'],['HandRight','RightFinger4'],
        ['HandRight','RightFinger5'])
        #,['ShoulderCenter','HipCenter'],['HipCenter','HipRight'],
        #['HipCenter','HipLeft'],
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            print(link[1])
            print(self.getWorldCoordinates())
            p=[x/self.data_norm*640 for x in self.getWorldCoordinates()[link[1]][0:2]]
            print(p)
            p.extend([x/self.data_norm*640 for x in self.getWorldCoordinates()[link[0]][0:2]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getWorldCoordinates().keys():
            p=[x/self.data_norm*640 for x in self.getWorldCoordinates()[node][0:2]]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

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
        skeletonPath = os.path.join(fileName, sampleID + '_light_hand.csv')

        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]

        df = pd.read_csv(skeletonPath, header=None)
        for index, row in df.iterrows():
            # sprint(row.shape)
            self.skeletons.append(Skeleton(row))

        labelsPath = os.path.join(fileName, sampleID + '_label.csv')
        self.numFrames = len(self.skeletons)

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
                gestureID, sframe,eframe = label_info.split(',')
                #print(gestureID,sframe,eframe)
                if int(gestureID) == 99:
                    gestureID = 32
                self.labels.append([int(gestureID)-1,int(sframe)+1,int(eframe)+1])
            # print(self.labels)

    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        # Check the given file

        if frameNum<1 or frameNum>self.numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(self.numFrames)))
        return self.skeletons[frameNum-1]

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.numFrames

    def getMGFlag(self):
        """ Get the number of frames for this sample """
        MG_flag = numpy.zeros( self.numFrames)
        for ges_info in self.labels:
            [gestureID, startFrame, endFrame] = ges_info
            # Get the gesture ID, and start and end frames for the gesture
            MG_flag[startFrame:endFrame] = 1
        return MG_flag


    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,640,(255,255,255))

    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels

    def exportPredictions(self, prediction,predPath):
        """ Export the given prediction to the correct file in the given predictions path """

        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath, self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'w')
        for row in prediction:
            output_file.write(str(int(row[0])) + "," + str(int(row[1])) + "," + str(int(row[2])) + "\n")
        output_file.close()
