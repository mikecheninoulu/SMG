import os
import sys
import pickle
import csv
import argparse
import numpy as np
import pandas as pd
import xlrd
from numpy.lib.format import open_memmap

from utils.mg_read_skeleton import read_xyz,read_xyz_gesture_long

max_body = 1
num_joint = 25
max_frame = 90
toolbar_width = 30

with open('hidden_test_index.csv', newline='') as f:
    reader = csv.reader(f)
    Subject_independent_hidden_test_index = list(reader)[0]
with open('hidden_train_index.csv', newline='') as f:
    reader = csv.reader(f)
    Subject_independent_hidden_train_index = list(reader)[0]
with open('normal_test_index.csv', newline='') as f:
    reader = csv.reader(f)
    Subject_independent_normal_test_index = list(reader)[0]
with open('normal_train_index.csv', newline='') as f:
    reader = csv.reader(f)
    Subject_independent_normal_train_index = list(reader)[0]
# print(Subject_independent_normal_test_index)

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            part='eval',state = 'hidden'):

    #list_len = 3699

    sample_list = os.listdir(data_path)

    sample_name_list = []
    sample_label_list = []

    if state == 'hidden':
        if part== 'train':
            subject_index_list = Subject_independent_hidden_train_index
        elif part== 'test':
            subject_index_list = Subject_independent_hidden_test_index
            # print(Subject_independent_hidden_test_index)
        else:
            print('part error')
    elif state == 'normal':
        if part== 'train':
            subject_index_list = Subject_independent_normal_train_index
        elif part== 'test':
            subject_index_list = Subject_independent_normal_test_index
            # print(Subject_independent_hidden_test_index)
        else:
            print('train error')
    else:
        print('wrong')

    sample_list_len = len(subject_index_list)
    # print('sample length')
    # print(sample_list_len)

    fp = open_memmap(
        '{}/{}_{}_data.npy'.format(out_path, part,state),
        dtype='float32',
        mode='w+',
        shape=(sample_list_len, 3, max_frame, num_joint, max_body))

    index_global = 0
    index_local = 0
    for sample in sample_list[:]:
        # print(sample)

        sample_name = sample
        sample_path = data_path + '/' + sample_name

        label_path = sample_path + '/' + sample_name + '_tflabels_new.csv'
        skeleton_path = sample_path + '/' + sample_name + '_skeleton.csv'
        skeleton_list = []
        with open(skeleton_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                skeleton_list.append(row)
        n_frames =  len(skeleton_list)

        with open(label_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if str(index_global) in subject_index_list:
                    # print('pass')
                    # print(row)
                    [emo_id, startf, endf, emo_label] = row
                    # print()
                    emo_label = int(emo_label)
                    sample_name_list.append(sample_name+str(index_local))
                    if state == 'normal':
                        sample_label_list.append(emo_label%10 - 1)
                    elif state == 'hidden':
                        sample_label_list.append(emo_label//10)
                    else:
                        print('wrong')

                    print_toolbar(index_local * 1.0 /sample_list_len,'({:>5}/{:<5}) Processing {:<5} data: '.format(index_local + 1,sample_list_len, part))
                    data = read_xyz(skeleton_list, st_frame = int(startf), ed_frame = int(endf), max_body=max_body, num_joint=num_joint)

                    fp[index_local, :, 0:data.shape[1], :, :] = data
                    index_local += 1
                    end_toolbar()

                index_global += 1

    with open('{}/{}_{}_label.pkl'.format(out_path, part,state), 'wb') as f:
        pickle.dump((sample_name_list, list(sample_label_list)), f)


    # print(index_global)
    # print(sample_label_list)
    # print(fp)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MG-RGB Data Converter.')
    parser.add_argument(
        '--data_path', default='/media/haoyu/TITAN/1_SMG/Welldone/')
    parser.add_argument('--out_folder', default='data/SMGskeleton_emotion/')
    #parser.add_argument('--label_data', default='tools/labels_20200831.csv')

    part = [ 'train','test']#,
    states = ['hidden', 'normal']

    arg = parser.parse_args()

    for p in part:
        print(p)
        for s in states:
            print(s)
            out_path = arg.out_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                part=p,state=s)
