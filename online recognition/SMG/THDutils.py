
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
import shutil
import gc
import copy
import numpy
import random
import cv2
from PIL import Image, ImageDraw
import os
from functools import partial
from scipy.ndimage.filters import gaussian_filter
import time
import pickle
import re
from sklearn import preprocessing
import scipy.io as sio
from Att_BiLSTM_training import test_att_BiLSTM
from keras.models import *
from statistics import mode
from scipy import interpolate
from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
from SMGaccessSample import GestureSample
from scipy.special import softmax


'''
load SMG training dataset
'''
def loader_SMG(parsers):

    # start counting tim
    time_tic = time.time()

    #counting feature number
    HMM_state_feature_count = 0
    n_HMM_state_feature_count = 0

    #how many joints are used
    njoints = parsers['njoints']

    used_joints = parsers['used_joints']

    #HMM temporal steps
    Time_step_NO = parsers['STATE_NUM']#3

    #feature dimension of the LSTM input
    featurenum = int((njoints*(njoints-1)/2 + njoints**2)*3)

    #get sample list
    Sample_list = sorted(os.listdir(parsers['data']))

    dictionaryNum = parsers['class_count']*parsers['STATE_NUM']+1


    '''pre-allocating the memory '''
    #gesture features
    Feature_all_states = numpy.zeros(shape=(400000, featurenum), dtype=numpy.float32)
    Targets_all_states = numpy.zeros(shape=(400000, dictionaryNum), dtype=numpy.uint8)

    n_Feature_all_states = numpy.zeros(shape=(400000, featurenum), dtype=numpy.float32)
    n_Targets_all_states = numpy.zeros(shape=(400000, dictionaryNum), dtype=numpy.uint8)

    # HMM pror and transition matrix
    Prior = numpy.zeros(shape=(dictionaryNum))
    Transition_matrix = numpy.zeros(shape=(dictionaryNum,dictionaryNum))

    #start traversing samples
    for sampleID in Sample_list[0:35]:

        print("\t Processing file " + str(sampleID))


        smp=GestureSample(os.path.join(parsers['data'],sampleID),sampleID)

        gesturesList=smp.getGestures()

        Transition_matrix[-1, -1] += smp.getNumFrames()

        MG_flag = smp.getMGFlag()

        '''traverse all ges in samples'''
        #get the skeletons of the gesture
        for ges_info in gesturesList:
            gestureID, startFrame, endFrame = ges_info
            # print(endFrame - startFrame)
            '''
            1. extract skeleton features of ges
            '''
            Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame)
            # to see we actually detect a skeleton:

             ### extract the features according to the CVPR2014 paper
            Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)

            '''generatre the corresponding labels'''
            sample_label = extract_HMMstate_label(Time_step_NO, Feature.shape[0], dictionaryNum, gestureID)

            #assign seg_length number of features to current gesutre
            Feature_all_states[HMM_state_feature_count:HMM_state_feature_count + Feature.shape[0], :] = Feature
            #assign seg_length number of labels to corresponding features
            Targets_all_states[HMM_state_feature_count:HMM_state_feature_count + Feature.shape[0], :] = sample_label
            #update feature count
            HMM_state_feature_count = HMM_state_feature_count + Feature.shape[0]


            '''
            2. extract skeleton features of non ges
            '''
            '''non movement before 5 frames'''
            if 1 not in MG_flag[startFrame-5:startFrame]:
                n_startFrame = startFrame-5
                n_endFrame = startFrame

                Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, n_startFrame, n_endFrame)

                Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)

                '''generatre the corresponding labels'''
                #generatre the corresponding labels
                sample_label = extract_nonActionlabel(Feature.shape[0], dictionaryNum)
                #assign seg_length number of features to current gesutre
                n_Feature_all_states[n_HMM_state_feature_count:n_HMM_state_feature_count + Feature.shape[0], :] = Feature
                #assign seg_length number of labels to corresponding features
                n_Targets_all_states[n_HMM_state_feature_count:n_HMM_state_feature_count + Feature.shape[0], :] = sample_label
                #update feature count
                n_HMM_state_feature_count = n_HMM_state_feature_count + Feature.shape[0]

            '''non movement after 5 frames'''

            if 1 not in MG_flag[endFrame:endFrame+5]:
                n_startFrame = endFrame
                n_endFrame = endFrame+5

                Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, n_startFrame, n_endFrame)
                Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)

                '''generatre the corresponding labels'''
                #generatre the corresponding labels
                sample_label = extract_nonActionlabel(Feature.shape[0], dictionaryNum)
                #assign seg_length number of features to current gesutre
                n_Feature_all_states[n_HMM_state_feature_count:n_HMM_state_feature_count + Feature.shape[0], :] = Feature
                #assign seg_length number of labels to corresponding features
                n_Targets_all_states[n_HMM_state_feature_count:n_HMM_state_feature_count + Feature.shape[0], :] = sample_label
                #update feature count
                n_HMM_state_feature_count = n_HMM_state_feature_count + Feature.shape[0]


            '''
            3. extract HMM transition info of this ges
            '''

            for frame in range(endFrame-startFrame+1-4):
                # print(gestureID)
                state_no_1,state_no_2 = HMMmatrix(gestureID, frame, endFrame-startFrame+1-4, Time_step_NO)
                # print(state_no_2)
                ## we allow first two states add together:
                Prior[state_no_1] += 1
                Transition_matrix[state_no_1, state_no_2] += 1
                Transition_matrix[-1, -1] -= 1
                if frame<2:
                    Transition_matrix[-1, state_no_1] += 1
                    Prior[-1] += 1
                if frame> (endFrame-startFrame+1-4-2):
                    Transition_matrix[state_no_2, -1] += 1
                    Prior[-1] += 1

    Feature_all_states = Feature_all_states[:HMM_state_feature_count, :]
    #assign seg_length number of labels to corresponding features
    Targets_all_states = Targets_all_states[:HMM_state_feature_count, :]

    print('gesture')
    print(HMM_state_feature_count)

    n_Feature_all_states = n_Feature_all_states[:n_HMM_state_feature_count, :]
    #assign seg_length number of labels to corresponding features
    n_Targets_all_states = n_Targets_all_states[:n_HMM_state_feature_count, :]
    print('non gesture')
    print(n_HMM_state_feature_count)

    num_samples = 20000
    idx = numpy.random.randint(0,len(n_Feature_all_states),size=(num_samples))
    # print(idx)
    n_Feature_all_states = n_Feature_all_states[idx]
    n_Targets_all_states = n_Targets_all_states[idx]
    # save the feature file:
    Feature_all_states = numpy.concatenate((Feature_all_states, n_Feature_all_states))
    Targets_all_states = numpy.concatenate((Targets_all_states, n_Targets_all_states))
    HMM_state_feature_count +=num_samples

    Feature_all,Targets_all,SK_normalizationfilename = process_feature(parsers,Feature_all_states,Targets_all_states,HMM_state_feature_count)

    print ("Processing data done with consuming time %d sec" % int(time.time() - time_tic))

    return Prior, Transition_matrix, Feature_all,Targets_all,SK_normalizationfilename


'''
sparse data list
'''
def process_feature(parsers, Feature_all,Targets_all,action_feature_count):

    # save the feature file:
    print ('total training samples: ' + str(action_feature_count))
    Feature_all = Feature_all[0:action_feature_count, :]
    Targets_all = Targets_all[0:action_feature_count, :]


    #random the samples
    rand_num = numpy.random.permutation(Feature_all.shape[0])
    Feature_all = Feature_all[rand_num]
    Targets_all  = Targets_all[rand_num]


    #[train_set_feature_normalized, Mean1, Std1]  = preprocessing.scale(train_set_feature)
    scaler = preprocessing.StandardScaler().fit(Feature_all)
    Mean1 = scaler.mean_
    Std1 = scaler.scale_
    Feature_all = normalize(Feature_all,Mean1,Std1)

    # save the normalization files
    SK_normalizationfilename = parsers['outpath']+ parsers['experi_ID'] +'SK_normalization.pkl'

    f = open(SK_normalizationfilename,'wb')
    pickle.dump( {"Mean1": Mean1, "Std1": Std1 },f)
    f.close()

    return Feature_all,Targets_all, SK_normalizationfilename

'''
using DNN to get HMM emission probability
'''
def emission_prob(modelname, parsers, Feature, Mean1, Std1,best_threshold,modeltype,AES=True,mu=0.0,lamd=1.0):
    Feature_normalized = normalize(Feature, Mean1, Std1)
    #print Feature_normalized.max()
    #print Feature_normalized.min()

    # feed to  Network
    y_pred_4 = test_att_BiLSTM(modelname, parsers, Feature_normalized, best_threshold)
    #y_pred_4 = test_torch_BiLSTM(modelname, parsers, Feature_normalized,best_threshold, modeltype, iflog)

    if AES:

        y_pred_4 = y_pred_4.T
        observ_likelihood = y_pred_4
        # print(y_pred_4)
        atten_value = y_pred_4[:-1, :]* y_pred_4[:-1, :]/9
        # print(atten_value)softmax(x, axis=0)
        SFMX = softmax(atten_value, axis=0)
        #
        # print(SFMX)
        observ_likelihood[:-1, :] = SFMX* y_pred_4[:-1, :]*mu + y_pred_4[:-1, :]
        # print(observ_likelihood)
        observ_likelihood= numpy.log(observ_likelihood)
        # print(observ_likelihood)
        observ_likelihood[-1, :] = observ_likelihood[-1, :] *lamd


    else:
        observ_likelihood = numpy.log(y_pred_4.T)

    return observ_likelihood

'''
save features
'''
def datasaver(parsers, Prior, Transition_matrix,Feature_all,Targets_all):

    '''define name'''

    Prior_Transition_matrix_filename = parsers['outpath']+ parsers['experi_ID'] + str(parsers['STATE_NUM']) + 'Prior_Transition_matrix.mat'
    Feature_filename = parsers['outpath'] + parsers['experi_ID'] + str(parsers['STATE_NUM']) + 'feature'+ str(parsers['featureNum']) +'Feature_all.pkl'

    '''HMM transition state'''
    #save HMM transition matrix
    sio.savemat( Prior_Transition_matrix_filename, {'Transition_matrix':Transition_matrix, 'Prior': Prior})

    '''feature storage'''
    # save the skeleton file:
    f = open(Feature_filename, 'wb')
    pickle.dump({"Feature_all": Feature_all, "Targets_all": Targets_all }, f,protocol=4)
    f.close()

    return Prior_Transition_matrix_filename, Feature_filename

'''
package Parameters into one file
'''
def packagePara(parsers, model1name, norm_name,HMM_file):

    Paras = {'model1':model1name,
             'norm_para':norm_name,
             'HMM_model':HMM_file,
             }
    path = parsers['outpath'] + parsers['experi_ID'] + '/Paras.pkl'
    afile = open(path, 'wb')
    pickle.dump(Paras, afile)
    afile.close()

'''
test result
'''
def tester_SMG(parsers,best_threshold):

    used_joints = parsers['used_joints']
    dictionaryNum = parsers['class_count']*parsers['STATE_NUM']+1

    MODEL4 = load_model(parsers['outpath']+ parsers['experi_ID'] +'/my_model.h5')
    #print(ges_info_list)
    correct_count = 0.0
    total_count = 0.0
    acc_total = 0.0

    time_tic = time.time()
    datacheck = []

    path = parsers['outpath']+ parsers['experi_ID'] + '/Paras.pkl'
    file2 = open(path, 'rb')
    Paras = pickle.load(file2)
    file2.close()

    ### load the pre-store normalization constant
    f = open(Paras['norm_para'],'rb')
    SK_normalization = pickle.load(f)
    Mean1 = SK_normalization ['Mean1']
    Std1 = SK_normalization['Std1']

    ## Load networks
    modelname1 = Paras['model1']

    ## Load Prior and transitional Matrix
    dic=sio.loadmat(Paras['HMM_model'])
    Transition_matrix = dic['Transition_matrix']

    Transition_matrix[-1, -1]= Transition_matrix[-1, -1]

    Prior = dic['Prior']
    ## Load trained networks
    njoints = parsers['njoints']

    #get sample list
    Sample_list = sorted(os.listdir(parsers['data']))

    total_F1 = 0.0
    total_acc = 0.0
    total_rec = 0.0

    recall_len = 0
    acc_len = 0
    correct_count = 0.0
    threshold_alpha =  parsers['threshold_alpha']
    #start traversing samples
    for sampleID in Sample_list[35:]:

        print(sampleID)
        time_single = time.time()

        '''1 extract skeleton features of this ges'''
        '''process the gesture parts'''
        smp=GestureSample(os.path.join(parsers['data'],sampleID),sampleID)

        gesturesList=smp.getGestures()

        frame_count =smp.getNumFrames()

        MG_flag = smp.getMGFlag()

        Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints,  1, frame_count - 1)
        # to see we actually detect a skeleton:

         ### extract the features according to the CVPR2014 paper
        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
        #ratio = 0.8
        #visiblenumber = int(ratio* (frame_count))

        # sample_feature1 = copy.copy(sample_feature)
        # observ_likelihood1 = emission_prob(modelname1, parsers, sample_feature1, Mean1, Std1, best_threshold, parsers['netmodel'],False)

        sample_feature2 = copy.copy(Feature)
        log_observ_likelihood1 = emission_prob(MODEL4, parsers, sample_feature2, Mean1, Std1, best_threshold, parsers['netmodel'], True,parsers['mu'],parsers['lambda'] )#parsers['AES'])

        log_observ_likelihood1[-1, 0:5] = 0
        log_observ_likelihood1[-1, -5:] = 0
        #
        log_observ_likelihood = log_observ_likelihood1#[:,mask]# + log_observ_likelihood1
        #print("\t Viterbi path decoding " )
        #do it in log space avoid numeric underflow
        [path, _, global_score] = viterbi_path_log(numpy.log(Prior), numpy.log(Transition_matrix), log_observ_likelihood)

        pred_label, begin_frame, end_frame, Individual_score, frame_length = viterbi_colab_MES(path, global_score, state_no = parsers['STATE_NUM'], threshold=-3, mini_frame=15, cls_num = parsers['class_count'])#viterbi_colab_clean_straight(parsers['STATE_NUM'], parsers['class_count'], path, global_score,)


        MG_flag = numpy.zeros(smp.getNumFrames())

        #
        # if True:
        #     # global_score=global_score[:,0:1000]
        #     im  = imdisplay(global_score)
        #     plt.imshow(im, cmap='Greys')## cmap='gray')
        #     plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
        #     plt.xlim((0, 2000))#global_score.shape[-1]))
        #     plt.ylim((0, 100))#global_score.shape[-2]))
        #     # plot ground truth
        #     for gesture in gesturesList:
        #     # Get the gesture ID, and start and end frames for the gesture
        #         gestureID,startFrame,endFrame=gesture
        #         frames_count = numpy.array(range(startFrame, endFrame+1))
        #         pred_label_temp = ((gestureID-1) *5+3) * numpy.ones(len(frames_count))
        #         plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
        #
        #     # plot clean path
        #     for i in range(len(begin_frame)):
        #         frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
        #         pred_label_temp = ((pred_label[i]-1) *5+3) * numpy.ones(len(frames_count))
        #         plt.plot(frames_count, pred_label_temp, color='b', linewidth=2.0)
        #
        #     plt.show()
        # #


        pred_len = len(pred_label)

        # if True:
        #     # fig = plt.figure()
        #     # fig.set_facecolor("antiquewhite")
        #     im  = imdisplay(global_score)
        #     plt.imshow(im, cmap = 'antiquewhite')
        #     plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
        #     plt.xlim((0, 1000))#global_score.shape[-1]))
        #     plt.ylim((0, 100))#global_score.shape[-2]))
        #     # plot ground truth
        #     for gesture in gesturesList[:10]:
        #     # Get the gesture ID, and start and end frames for the gesture
        #         gestureID,startFrame,endFrame=gesture
        #         frames_count = numpy.array(range(startFrame, endFrame+1))
        #         pred_label_temp = ((gestureID-1) *10 +5) * numpy.ones(len(frames_count))
        #         plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
        #
        #     # plot clean path
        #     for i in range(len(begin_frame)):
        #         frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
        #         pred_label_temp = ((pred_label[i]-1) *10 +5) * numpy.ones(len(frames_count))
        #         plt.plot(frames_count, pred_label_temp, color='b', linewidth=2.0)
        #
        #     plt.show()


        current_correct_count = 0.0

        for ges in gesturesList:
            gt_label, gt_begin_frame, gt_end_frame = ges
            for pred_i in range(len(begin_frame)):
                # print(pred_i)
                if pred_label[pred_i] == gt_label:
                    alpha = (min(end_frame[pred_i],gt_end_frame)-max(begin_frame[pred_i],gt_begin_frame))/(max(end_frame[pred_i],gt_end_frame)-min(begin_frame[pred_i],gt_begin_frame))
                    # print(pred_i)
                    if alpha> threshold_alpha:
                        # print(pred_label)
                        numpy.delete(pred_label, pred_i)
                        numpy.delete(begin_frame, pred_i)
                        numpy.delete(end_frame, pred_i)
                        #print(pred_label)
                        correct_count +=1
                        current_correct_count +=1
                        break

        recall_len =recall_len + len(gesturesList)
        acc_len = acc_len + pred_len
        print(len(gesturesList))
        print(pred_len)
        # pred_label
        '''recall'''
        current_recall = current_correct_count/len(gesturesList)
        '''precise'''
        current_precision = current_correct_count/pred_len
        current_F1_score = 2*current_recall* current_precision/(current_recall + current_precision+0.0000001)
        print("Used time %d sec, processing speed %f fps, F1 score%f, Recall %f, precision%f" %(int(time.time() - time_single),frame_count/float(time.time() - time_single),current_F1_score, current_recall, current_precision))



    '''recall'''
    recall = correct_count/recall_len
    '''precise'''
    precision = correct_count/acc_len


    total_acc += precision
    total_rec += recall
    F1_score = 2*recall* precision/(recall + precision+0.0000001)
    total_F1 += F1_score

    print ("Processing testing data done with consuming time %d sec" % int(time.time() - time_tic))

    #F1_total = total_F1/len(Sample_list[35:])
    print(parsers['experi_ID']+". The rec for this prediction is " + "{:.12f}".format(total_rec))
    print(parsers['experi_ID']+". The acc for this prediction is " + "{:.12f}".format(total_acc))
    print(parsers['experi_ID']+". The score for this prediction is " + "{:.12f}".format(total_F1))
    print('mu'+str(parsers['mu'])+'lambda'+str(parsers['lambda']))
    numpy.savetxt(parsers['outpath']+ parsers['experi_ID'] +'_score_'+ str(acc_total) +'.txt', [])

def extract_temporal_movingPose(skeleton_all, frame_compens , LSTM_step, njoints):

    frame_count =  skeleton_all.shape[0] - frame_compens*2

    feature_all = Extract_moving_pose_Feature(skeleton_all, njoints)

    feature_dim = feature_all.shape[1]

    feature_n  = numpy.zeros(shape=(frame_count, feature_dim*LSTM_step))

    for frame in range(frame_count):

        for step in range(LSTM_step):

            feature_n[frame, step*feature_dim:(step+1)*feature_dim] = feature_all[frame+step,:]

    return feature_n, feature_n.shape[0]


def extract_HMMstate_label(STATE_NO, action_count, dictionaryNum, gestureID):
    # label the features
    target = numpy.zeros(shape=(action_count, dictionaryNum))
    # HMM states force alignment
    for i in range(STATE_NO):
        # get feature index of the current time step
        begin_feature_index = int(numpy.round(action_count * i / STATE_NO) + 1)
        end_feature_index = int(numpy.round(action_count * (i + 1) / STATE_NO))
        # get feature length of the current time step
        seg_length = end_feature_index - begin_feature_index + 1
        labels = numpy.zeros(shape=(dictionaryNum, 1))
        # assign the one hot labels
        try:
            labels[ i + STATE_NO*gestureID] = 1
        except:
            print(labels.shape)
            print( i + STATE_NO*gestureID)

        target[begin_feature_index-1:end_feature_index,:] = numpy.tile(labels.T, (seg_length, 1))
    return target

def extract_nonActionlabel (action_count, dictionaryNum):
    target_n = numpy.zeros(shape=(action_count, dictionaryNum))
    target_n[:,-1] = 1
    return target_n

'''
record HMM transition matrix
'''
def HMMmatrix(gestureID, frame, frame_count,STATE_NO):
    state_no_1 = numpy.floor(STATE_NO*(frame*1.0/(frame_count+3)))
    state_no_1 = int(state_no_1+STATE_NO*(gestureID))
    state_no_2 = numpy.floor(STATE_NO*((frame+1)*1.0/(frame_count+3)))
    state_no_2 = int(state_no_2+STATE_NO*(gestureID))
    return state_no_1,state_no_2


def normalize(Data, Mean, Std):
    # print(Data.shape)
    # print(Mean.shape)
    Data -= Mean
    Data /= Std
    return Data


def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame):
    """
    Extract original features
    """
    frame_num = 0
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))

    for numFrame in range(startFrame,endFrame+1):
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            # print((skel.joins[used_joints[joints]][0]))
            # print(Skeleton_matrix[frame_num, joints*3: (joints+1)*3])
            print(skel.joins[used_joints[joints]][0])
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] =skel.joins[used_joints[joints]][0]

        frame_num += 1


    if numpy.allclose(sum(sum(numpy.abs(Skeleton_matrix))),0):
        valid_skel = False
    else:
        valid_skel = True

    return Skeleton_matrix, valid_skel



def Extract_moving_pose_Feature(Skeleton_matrix_Normalized, njoints):

    #pose
    F_pose = Skeleton_matrix_Normalized

    #velocity
    F_velocity = Skeleton_matrix_Normalized[2:,:] - Skeleton_matrix_Normalized[0:-2,:]

    #accelerate
    F_accelerate = Skeleton_matrix_Normalized[4:,:] + Skeleton_matrix_Normalized[0:-4,:] - 2 * Skeleton_matrix_Normalized[2:-2,:]

    #absolute pose
    FeatureNum = 0
    F_abs = numpy.zeros(shape=(Skeleton_matrix_Normalized.shape[0], int(njoints * (njoints-1)/2)))

    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            all_X = Skeleton_matrix_Normalized[:, joints1*3] - Skeleton_matrix_Normalized[:, joints2*3]
            all_Y = Skeleton_matrix_Normalized[:, joints1*3+1] - Skeleton_matrix_Normalized[:, joints2*3+1]
            all_Z = Skeleton_matrix_Normalized[:, joints1*3+2] - Skeleton_matrix_Normalized[:, joints2*3+2]

            Abs_distance = numpy.sqrt(all_X**2 + all_Y**2 + all_Z**2)
            F_abs[:, FeatureNum] = Abs_distance
            FeatureNum += 1

    Features = numpy.concatenate((F_pose[2:-2, :], F_velocity[1:-1,:], F_accelerate, F_abs[2:-2, :]), axis = 1)
    return Features


def viterbi_path_log(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] =  observ_likelihood[:, 0]
    # print(global_score.shape)
    # need to  normalize the data

    for t in range(1, T):
        for j in range(N):
            # print(global_score[:, t-1].shape)
            # print(prior.shape)

            temp = global_score[:, t-1] + transmat[:, j] + observ_likelihood[j, t]-prior[0]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

    path[T-1] = global_score[:, T-1].argmax()

    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]


    return [path, predecessor_state_index, global_score]


def viterbi_colab_MES(path, global_score, state_no = 5, threshold=-3, mini_frame=15, cls_num = 10):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """
    # just to accommodate some frame didn't start right from the begining
    all_label = state_no * cls_num # 20 vocabularies
    start_label = numpy.concatenate((range(0,all_label,state_no), range(1,all_label,state_no),range(2,all_label,state_no)))
    end_label   = numpy.concatenate((range(state_no-3,all_label,state_no), range(state_no-2,all_label,state_no),range(state_no-1,all_label,state_no)))


    begin_frame = []
    end_frame = []
    pred_label = []

    frame = 1
    while(frame < path.shape[-1]-1):
        if path[frame-1]==all_label and path[frame] in start_label:
            begin_frame.append(frame)
            # python integer divsion will do the floor for us :)
            pred_label.append( int(path[frame]/state_no))
            while(frame < path.shape[-1]-1):
                if path[frame] in end_label and path[frame+1]==all_label:
                    end_frame.append(frame)
                    break
                else:
                    frame += 1
        frame += 1

    end_frame = numpy.array(end_frame)
    begin_frame = numpy.array(begin_frame)
    pred_label= numpy.array(pred_label)


    if len(begin_frame)> len(end_frame):
        begin_frame = begin_frame[:-1]
        pred_label = pred_label[:-1]

    elif len(begin_frame)< len(end_frame):# risky hack! just for validation file 668
        end_frame = end_frame[1:]
    ## First delete the predicted gesture less than 15 frames
    frame_length = end_frame - begin_frame
    ## now we delete the gesture outside the vocabulary by choosing
    ## frame number small than mini_frame
    mask = frame_length > mini_frame
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]


    Individual_score = []
    for idx, g in enumerate(begin_frame):
            score_start = global_score[path[g], g]
            score_end = global_score[path[end_frame[idx]], end_frame[idx]]
            Individual_score.append(score_end - score_start)

    ## now we delete the gesture outside the vocabulary by choosing
    ## score lower than a threshold
    Individual_score = numpy.array(Individual_score)
    frame_length = end_frame - begin_frame
    # should be length independent
    Individual_score = Individual_score/frame_length

    order = Individual_score.argsort()
    ranks = order.argsort()

    mask = Individual_score > threshold
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]
    Individual_score = Individual_score[mask]


    return [pred_label, begin_frame, end_frame, Individual_score, frame_length]

def Extract_feature_Realtime(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], int(njoints * (njoints-1)/2*3)))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1

    #F_cp
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-1, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[1:,joints1*3:(joints1+1)*3]-Pose[0:-1,joints2*3:(joints2+1)*3]
            FeatureNum += 1

    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf

    Features = numpy.concatenate( (Fcc[0:-1, :], Fcp), axis = 1)
    return Features




def imdisplay(im):
    """ display grayscale images
    """
    im_min = im.min()
    im_max = im.max()
    return (im - im_min) / (im_max -im_min)
