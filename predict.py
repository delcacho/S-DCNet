import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torch.optim as optim
import torchvision
import cv2
#from torchvision import models
import torchvision.transforms.functional as TF

import os
import numpy as np
from time import time

import math
import pandas as pd
import csv

from IOtools import txt_write
from load_data_V2 import myDataset, ToTensor

from Network.SDCNet import SDCNet_VGG16_classify
from Val import test_phase
from IOtools import txt_write
from load_data_V2 import get_pad

def openCvToTensor(img):
   imgtensor = torch.from_numpy(img/255.)
   imgtensor = imgtensor.permute(2,0,1)
   index = torch.LongTensor([2, 1, 0])
   y = torch.zeros_like(imgtensor)
   y[index] = imgtensor
   return(y)

def predictTensor(imgtensor):
   inputs = get_pad(imgtensor)
   inputs = inputs.type(torch.float32)
   features = net(inputs[None, ...])
   div_res = net.resample(features)
   merge_res = net.parse_merge(div_res)
   outputs = merge_res['div'+str(net.args['div_times'])]
   del merge_res
   pre =  (outputs).sum()
   result = pre.double().item()
   return(result)

if __name__ == '__main__':

    cap=cv2.VideoCapture("https://video-auth1.iol.pt/beachcam/tocha/playlist.m3u8")
    ret, img = cap.read()
    #convert to RGB from BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./beachtest/test/prueba.png",img)
    opt = dict()
    model_list = {0:'model/SHA',1:'model/SHB'}
    max_num_list = {0:22,1:7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1 
    opt = dict()

    model_list = {0:'model/SHA',1:'model/SHB'}
    max_num_list = {0:22,1:7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1

    # --Network settinng    
    opt['psize'],opt['pstride'] = 64,64

    
    # -- start testing
    set_len = 2

    for ti in range(set_len):
        opt['trained_model_path'] = model_list[ti]

        #-- set the max number and partition
        opt['max_num'] = max_num_list[ti]  
        partition_method = {0:'one_linear',1:'two_linear'}
        opt['partition'] = partition_method[1]
        opt['step'] = 0.5

        print('=='*36)
        print('Begin to test for %s' %(model_list[ti]) )

        # =============================================================================
        # inital setting
        # =============================================================================
        # 1.Initial setting
        # --1.1 dataset setting
        num_workers = opt['num_workers']

        img_subsubdir = 'images'; tar_subsubdir = 'gtdens'
        dataset_transform = ToTensor()

        # --1.2 use initial setting to generate
        # set label_indice
        if opt['partition'] =='one_linear':
            label_indice = np.arange(opt['step'],opt['max_num']+opt['step']/2,opt['step'])
            add = np.array([1e-6])
            label_indice = np.concatenate( (add,label_indice) )
        elif opt['partition'] =='two_linear':
            label_indice = np.arange(opt['step'],opt['max_num']+opt['step']/2,opt['step'])
            add = np.array([1e-6,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]) 
            label_indice = np.concatenate( (add,label_indice) )
        # print(label_indice)

        opt['label_indice'] = label_indice
        opt['class_num'] = label_indice.size+1

        #test settings
        testloader = torchvision.datasets.ImageFolder(
            root="./beachtest",
            transform=torchvision.transforms.ToTensor()
        )
        print(testloader.imgs)

        # init networks
        label_indice = torch.Tensor(label_indice)
        class_num = len(label_indice)+1
        div_times = 2
        net = SDCNet_VGG16_classify(class_num,label_indice,psize=opt['psize'],\
            pstride = opt['pstride'],div_times=div_times,load_weights=True)#.cuda()

        # test the exist trained model
        mod_path='best_epoch.pth' 
        mod_path=os.path.join(opt['trained_model_path'],mod_path)

        if os.path.exists(mod_path):
            all_state_dict = torch.load(mod_path,map_location=torch.device('cpu'))
            net.load_state_dict(all_state_dict['net_state_dict'])
            log_save_path = os.path.join(opt['trained_model_path'],'log-trained-model.txt')
            # test
            with torch.no_grad():
                net.eval()
                start = time()
                avg_frame_rate = 0
                for j, data in enumerate(testloader):
                    y = openCvToTensor(img)
                    count = predictTensor(y)
                    print(count)
                    count = predictTensor(data[0])
                    print(count)

