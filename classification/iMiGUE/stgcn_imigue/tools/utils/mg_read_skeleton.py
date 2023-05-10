import numpy as np
import os
import csv
import xlrd

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(skeleton_list, st_frame, ed_frame, max_body=2, num_joint=25, max_frame = 90):
    # seq_info = read_skeleton(file)

    all_length = len(skeleton_list)

    st_frame = int(st_frame)
    ed_frame = int(ed_frame)
    length = int(ed_frame)-int(st_frame)

    if length > max_frame:
        length = max_frame
    data = np.zeros((3, length, num_joint, max_body))

    for frame in range(length):
        for j in range(num_joint):
            skeleton_frame = skeleton_list[st_frame + frame]
            #print(skeleton_frame)
            #print(a)
            data[:, frame, j, 0] = [skeleton_frame[0 + 11*j], skeleton_frame[1 + 11*j], skeleton_frame[2 + 11*j]]
    return data

def read_xyz_gesture_light(sh, st_frame, ed_frame, max_body=2, num_joint=25, max_frame = 90,n_frames=800):
    # seq_info = read_skeleton(file)

    st_frame = int(st_frame)
    ed_frame = int(ed_frame)
    length = int(ed_frame)-int(st_frame)+1

    #intrival = int(length*1.0/max_frame)

    #sample_range = range(0,length,intrival)

    data = np.zeros((3, length, num_joint, max_body))

    for frame in range(length):
        for j in range(num_joint):
            print(len(sh))
            skeleton_frame = sh[st_frame + frame]
            # print(skeleton_frame)
            data[:, frame, j, 0] = [skeleton_frame[0 + 3*j], skeleton_frame[1 + 3*j], skeleton_frame[2 + 3*j]]

    # print(data[:, 0, :, 0])
    return data


def read_wl_gesture(sh, max_body=2, num_joint=25, max_frame = 90):
    # seq_info = read_skeleton(file)
    length = len(sh)
    data = np.zeros((3, max_frame, num_joint, max_body))

    for frame in range(length):
        for j in range(num_joint):
            skeleton_frame = sh[frame]
            data[:, frame, j, 0] = [skeleton_frame[0 + 3*j], skeleton_frame[1 + 3*j], skeleton_frame[2 + 3*j]]
    return data
