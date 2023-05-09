#-------------------------------------------------------------------------------
# Name:        THD-HMM for gesture recognition
# Purpose:     The main script to run the project
# Copyright (C) 2018 Haoyu Chen <chenhaoyucc@icloud.com>,
# author: Chen Haoyu
# @center of Machine Vision and Signal Analysis,
# Department of Computer Science and Engineering,
# University of Oulu, Oulu, 90570, Finland
#-------------------------------------------------------------------------------

#from BiRNN_Training import train_BiLSTM,BiRNN
from THDutils import loader_SMG, datasaver, packagePara, tester_SMG
from Att_BiLSTM_training import train_att_BiLSTM

import os, shutil
#from keras.models import load_model

PC = 'chen'
if PC == 'chen':
    outpath = './experi/'
if PC == 'csc':
    outpath = '/wrk/chaoyu/DHori_experidata/'

if __name__ == '__main__':
    parsers = {
            'STATE_NUM' :5,
            'training_steps':40,
            'outpath':outpath,
            'Trainingmode':True,
            'Testingmode': True,
            'njoints':25,
            # Data folder (Training data)
            #'data': "./OADdataset/data/",
            'data': "./experimentWell_skeleton/",
            'MovingPose':True,
            #used joint list
            #'used_joints': ['HipCenter', 'ShoulderCenter', 'Head',
            #   'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
            #   'ShoulderRight','ElbowRight','WristRight','HandRight'],
            'used_joints':['SpineBase','SpineMid','ShoulderCenter','Head','ShoulderLeft','ElbowLeft','WristLeft','HandLeft','ShoulderRight',
            'ElbowRight','WristRight','HandRight','HipLeft','KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight',
            'SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight'],
            # 'class_count' : 10,# for OAD
            'class_count' : 17, # for SMG
            'maxframe':150,
            'AES':True,
            'mu':-2.5,
            'lambda':2.1,
            # get total state number of the THD model
            # Training Parameters
            'netmodel':2,
            'batch_size':32,
            'LSTM_step' : 5, # should be an odd number
            'njoints':25,
            'threshold_alpha':0.3,
            # get total state number of the THD model
            'featureNum' : 525}

    if parsers['Trainingmode']:
        parsers['experi_ID'] = 'state' + str(parsers['STATE_NUM']) + '_model' + str(parsers['netmodel']) + '_epoch'+ str(parsers['training_steps'])+'/'
        expripath = parsers['outpath'] + parsers['experi_ID']
        print(expripath)
        if os.path.exists(expripath):
            shutil.rmtree(expripath)
        os.mkdir(expripath)
        mode = 'train'
        #
        Prior, Transition_matrix,Feature_all,Targets_all,norm_name = loader_SMG(parsers)

        HMM_file_train, train_Feature_file = datasaver(parsers, Prior, Transition_matrix,Feature_all,Targets_all)

        # HMM_file = 'flat10Prior_Transition_matrix_complete.mat'
        # Feature_file = 'flat10feature154Feature_all_complete.pkl'
        # norm_name = 'origstep_10_state201_fea_154SK_normalization_complete.pkl'

        model1name, best_threshold = train_att_BiLSTM(parsers, train_Feature_file)
        #model1name,best_threshold = train_torch_BiLSTM(parsers, train_Feature_file, parsers['netmodel'])
        #model1name = 'snapshot_acc_96.8750_loss_0.123691_iter_51000_model.pt'
        model2name = []
        packagePara(parsers,model1name,norm_name,HMM_file_train)

    if parsers['Testingmode']:
        parsers['experi_ID'] = 'state' + str(parsers['STATE_NUM']) + '_model' + str(parsers['netmodel']) + '_epoch'+ str(parsers['training_steps'])+'/'
        expripath = parsers['outpath'] + parsers['experi_ID']
        best_threshold = 0
        tester_SMG(parsers,best_threshold)
