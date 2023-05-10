import os
import sys
import pickle
import csv
import argparse
import numpy as np
import pandas as pd
import xlrd
from tqdm import tqdm
from numpy.lib.format import open_memmap
from iMiGUEaccessSample import GestureSample

from utils.mg_read_skeleton import read_xyz,read_wl_gesture


used_joints = ['Nose','ShoulderCenter','ShoulderLeft','ElbowLeft','HandLeft','ShoulderRight','ElbowRight','HandRight','EyeLeft',
'EyeRight','EarLeft','EarRight', 'RightFinger1', 'RightFinger2', 'RightFinger3', 'RightFinger4', 'RightFinger5',
'LeftFinger1','LeftFinger2','LeftFinger3','LeftFinger4','LeftFinger5']
max_body = 1
num_joint = len(used_joints)
max_frame = 90
toolbar_width = 30

def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame, train_mode=True):
    """
    Extract original skeleton data
    """
    valid_skel = True
    frame_num = 0
    Skeleton_matrix  = np.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for numFrame in range(startFrame,endFrame+1):
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        if train_mode:
            if int(sum(skel.joins['HandRight']))== 0 or int(sum(skel.joins['HandLeft'])) ==0:
                #print('bad sample')
                valid_skel = False
                break

        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3] = skel.joins[used_joints[joints]][0]
            Skeleton_matrix[frame_num, joints*3+1] = skel.joins[used_joints[joints]][1]
            Skeleton_matrix[frame_num, joints*3+2] = skel.joins[used_joints[joints]][2]
        frame_num += 1

    if np.allclose(sum(sum(np.abs(Skeleton_matrix))),0):
        valid_skel = False
    else:
        valid_skel = True

    return Skeleton_matrix, valid_skel

def gendata(data_path,
            out_path,
            part='eval'):

    sample_name_list = []
    sample_label_list = []

    print('preprocessing the samples')
    video_list = os.listdir(data_path)

    sample_list_len = 0
    for sampleID in tqdm(video_list):
        # if int(video) in bad_samples:
        #     continue
        fileName = os.path.join(data_path,sampleID)
        labelsPath = os.path.join(fileName, sampleID + '_label.csv')

        if not os.path.exists(labelsPath):
            warnings.warn("Labels are not available", Warning)
            labels=[]
        else:
            f = open(labelsPath, "r")
            label_List = f.read().splitlines()
            # print(label_List)

        inx = 0
        sample_list_len += len(label_List)

    #print(max(sample_label_list))
    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(sample_list_len, 3, max_frame, num_joint, max_body))

    print('total sample length')
    print(sample_list_len)
    index = 0

    sample_name_list = []
    sample_label_list = []
    video_list = os.listdir(data_path)

    print('saving the samples into pack')
    inx = 0
    for sampleID in tqdm(video_list):
        # if int(video) in bad_samples:
        #     continue
        smp=GestureSample(os.path.join(data_path,sampleID),sampleID)

        gesturesList=smp.getGestures()

        # print(sampleID)

        inx_sample = 0
        for ges_info in gesturesList:
            gestureID, startFrame, endFrame = ges_info

            # print(action)
            st_frame = int(startFrame)
            ed_frame = int(endFrame)
            label = int(gestureID)



            if (ed_frame - st_frame)> (max_frame-1):
                st_frame = ed_frame-max_frame+1
            if (ed_frame-st_frame) < 1:
                print(sampleID)
                print(inx)
            #print(inx)
            #print(sampleID+str(inx))
            sample_name_list.append(sampleID+'_'+str(inx_sample))
            sample_label_list.append(label)
            sum_skele = 0.0

            skeleton_list, valid_skel = Extract_feature_UNnormalized(smp, used_joints,  st_frame, ed_frame, False)

            # note that we only use upper body joints !!!!
            skeleton_list= list(skeleton_list)

            # print(skeleton_list.shape)

            data = read_wl_gesture(skeleton_list, max_body=max_body, num_joint=num_joint,max_frame = max_frame)
            # print(sum([sum(i) for i in data]))
            # sum_skele = sum(sum([sum(i) for i in data]))

            # print(sum_skele[0])
            #
            # if sum_skele == 0.0:
            #     print('bad')

            fp[inx, :, 0:data.shape[1], :, :] = data


            inx +=1
            inx_sample +=1
        del smp

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name_list, list(sample_label_list)), f)

    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='iMiGUE-RGB Data Converter.')
    parser.add_argument('--data_path', default='./data/iMiGUE/')
    parser.add_argument('--out_folder', default='./data/iMiGUE/imigue_processed/')

    part = ['train','valid']
    arg = parser.parse_args()

    for p in part:

        print('processing: ' + p)

        out_path = arg.out_folder

        if p == 'train':
            data_path = arg.data_path + 'imigue_skeleton_train'
        elif p == 'valid':
            data_path = arg.data_path + 'imigue_skeleton_validate'
        else:
             print('error')

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gendata(
            data_path,
            out_path,
            part=p)
