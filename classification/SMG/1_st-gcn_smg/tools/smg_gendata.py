import os
import sys
import pickle
import csv
import argparse
import numpy as np
import pandas as pd
import xlrd
from numpy.lib.format import open_memmap

from utils.mg_read_skeleton import read_xyz,read_xyz_gesture

max_body = 1
num_joint = 25
max_frame = 60
toolbar_width = 30

subject_all = list(range(1,41))
# print(subject_all)
#subject_test = [8,11, 26, 56, 64, 16, 19, 60, 67, 40, 50, 9, 34, 37, 25, 47, 53, 36, 30, 33, 54, 68, 69, 4]
subject_test =[36,37,38,39,40]

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
            part='eval'):

    #list_len = 3699

    sample_list = os.listdir(data_path)

    sample_name_list = []
    sample_label_list = []

    for sample in subject_all:#[0:2]:
        # print(sample)
        id_list = []
        startf_list = []
        endf_list = []

        sample_name = 'Sample' + "%04d"%sample
        sample_path = data_path + '/' + sample_name

        label_path = sample_path + '/' + sample_name + '_finelabels.csv'
        skeleton_path = sample_path + '/' + sample_name + '_skeleton.csv'

        with open(label_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                [label, startf, endf] = row
                id_list.append(label)
                startf_list.append(startf)
                endf_list.append(endf)

        # print(len(action_list))

        istest = (sample in subject_test)

        if part == 'train':
            issample = not (istest)
            #print(issample)
        elif part == 'test':
            issample = istest
            #print(issample)
        else:
            raise ValueError()

        if issample:
            for inx in range(len(id_list)):
                # print(action)
                st_frame = int(startf_list[inx])
                ed_frame = int(endf_list[inx])
                label = int(id_list[inx])

                if (ed_frame-st_frame) < 1:
                    print(sample)
                    print(inx)
                sample_name_list.append(sample_name+str(inx))
                sample_label_list.append(label - 1)

    sample_list_len = len(sample_label_list)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(sample_list_len, 3, max_frame, num_joint, max_body))

    print(sample_list_len)
    index = 0

    sample_name_list = []
    sample_label_list = []
    for sample in subject_all:#[0:2]:
        # print(sample)

        id_list = []
        startf_list = []
        endf_list = []

        sample_name = 'Sample' + "%04d"%sample
        sample_path = data_path + '/' + sample_name

        label_path = sample_path + '/' + sample_name + '_finelabels.csv'
        skeleton_path = sample_path + '/' + sample_name + '_skeleton.csv'

        with open(label_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                [label, startf, endf] = row
                id_list.append(label)
                startf_list.append(startf)
                endf_list.append(endf)

        # print(len(action_list))

        istest = (sample in subject_test)

        if part == 'train':
            issample = not (istest)
            #print(issample)
        elif part == 'test':
            issample = istest
            #print(issample)
        else:
            raise ValueError()

        if issample:
            skeleton_list = []

            with open(skeleton_path, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    skeleton_list.append(row)

            n_frames =  len(skeleton_list)
            #print(len(skeleton_list[0]))

            for inx in range(len(id_list)):
                # print(action)
                st_frame = int(startf_list[inx])
                ed_frame = int(endf_list[inx])
                label = int(id_list[inx])

                if (ed_frame - st_frame)> (max_frame-1):
                    st_frame = ed_frame-max_frame+1
                if (ed_frame-st_frame) < 1:
                    print(sample)
                    print(inx)
                sample_name_list.append(sample_name+str(inx))
                sample_label_list.append(label - 1)
                sum_skele = 0.0
                print_toolbar(index * 1.0 /sample_list_len,'({:>5}/{:<5}) Processing {:<5} data: '.format(index + 1,sample_list_len, part))
                data = read_xyz(skeleton_list, st_frame = st_frame, ed_frame = ed_frame, max_body=max_body, num_joint=num_joint)
                # print(sum([sum(i) for i in data]))
                # sum_skele = sum(sum([sum(i) for i in data]))

                # print(sum_skele[0])
                #
                # if sum_skele == 0.0:
                #     print('bad')

                fp[index, :, 0:data.shape[1], :, :] = data
                end_toolbar()

                index = index + 1

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name_list, list(sample_label_list)), f)

    print()
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MG-RGB Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/haoyu/Documents/1_SMG/dataset/experimentWell')
    parser.add_argument('--out_folder', default='data/SMGskeleton/')
    #parser.add_argument('--label_data', default='tools/labels_20200831.csv')

    part = ['train', 'test']
    arg = parser.parse_args()

    for p in part:
        out_path = arg.out_folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gendata(
            arg.data_path,
            out_path,
            part=p)
