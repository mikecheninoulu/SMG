#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:33:07 2019

@author: haoyu

"""

import torch 
import torch.nn as nn

import numpy
import matplotlib.pyplot as plt



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PC = 'chen'
if PC == 'chen':
    outpath = './experi/'
if PC == 'csc':
    outpath = '/wrk/chaoyu/DHori_experidata/'
    
parsers = {
        'fusion_experi_ID':'mapping_from_5_to_10_onelayer',
        'outpath':outpath,
        # get gesture class number
        'class_count' : 20}
#
#class MLP_Fusion_one(nn.Module):
#    def __init__(self, input_size, output_size):
#        super(MLP_Fusion_one,self).__init__()
#        self.pro = nn.Linear(input_size, output_size)
#        
#    def forward(self,x):
#        dout = self.pro(x)
#        return dout
#    
#modelname = parsers['outpath']+ parsers['fusion_experi_ID'] +'/model_HMM_fusion.ckpt'
##Target_all_numeric = numpy.argmax(Target_all, axis=1)
#
##featurelen = parsers['featureNum']        
#input_state = 10
#        
#outputstate = 5
#
## add non gesture state
#dictionaryNum = parsers['class_count']*outputstate
#
#
#num_classes = dictionaryNum
#input_size = parsers['class_count']*input_state
#
#
#model = MLP_Fusion_one(input_size, num_classes).to(device)
#
#model.load_state_dict(torch.load(modelname))
#model.eval()
#with torch.no_grad():
#    weightvalue = model.pro.weight
#
#mapping_matrix = weightvalue.cpu().detach().numpy()
#print(weight)
import cPickle
import os

weight_file = os.path.join(parsers['outpath'],parsers['fusion_experi_ID'], 'Weight_all.pkl')

print ('... loading data')

f = file(weight_file,'rb' )
weight_file = cPickle.load(f)
f.close()

#READ FEATURES
init_weight = weight_file['weight_1']
mid1_weight = weight_file['weight_2']
mid2_weight = weight_file['weight_3']
mid3_weight = weight_file['weight_4']
best_weight = weight_file['weight_5']


#init_weight = list(reversed(init_weight))
#mid1_weight = list(reversed(mid1_weight))
#mid2_weight = list(reversed(mid2_weight))
#mid3_weight = list(reversed(mid3_weight))
#best_weight = list(reversed(best_weight))

#mask = best_weight > 0
#best_weight = best_weight[0:20,0:40]

fig, ax = plt.subplots()

ax.matshow(init_weight, cmap=plt.cm.Blues)

fig, ax = plt.subplots()

ax.matshow(mid1_weight, cmap=plt.cm.Blues)

fig, ax = plt.subplots()

ax.matshow(mid2_weight, cmap=plt.cm.Blues)
fig, ax = plt.subplots()

ax.matshow(mid3_weight, cmap=plt.cm.Blues)

fig, ax = plt.subplots()

ax.matshow(best_weight, cmap=plt.cm.Blues)

#for i in range(input_size):
#    print(i)
#    for j in range(num_classes):
#        c = mapping_matrix[i,j]
#        ax.text(i, j, str(c), va='center', ha='center')