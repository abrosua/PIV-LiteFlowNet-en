import sys, os
sys.path.append('/home/faber/thesis/hui-liteflownet/python')
import caffe
from caffe.proto import caffe_pb2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from caffenet import CaffeNet
from torch.autograd import Variable
from collections import OrderedDict
import csv

from caffenet import *


# Forwarding caffe with pytorch
def forward_pytorch(protofile, caffemodel, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(caffemodel)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models


# Model converter
def convert_pytorch(protofile, caffemodel):
    net = CaffeNet(protofile)
    # print(net)
    net.load_weights(caffemodel)

    return net


# Write layers name in csv
def layer_csv(title, param_dict):
    with open(title, 'w') as f:
        f.write("layers,params_shape\n")
        for key in param_dict.keys():
            f.write("%s,%s\n"%(key, str(param_dict[key])))


# Nvidia caffe converter (MANUAL Labor)
def nvidia_convert(protofile, caffemodel, save_title=True):
    net = caffe.Net(protofile, caffemodel, caffe.TEST)
    param_list = list(net.params.items())
    print(len(param_list))

    count = 0
    wnb = OrderedDict()
    misc, wnb_title = {}, {}
    # weights, biases, misc = {}, {}, {}
    for k, v in param_list:
        if len(v) > 2:
            val = []
            for item in v:
                store_val = np.array(item.data).reshape(item.data.shape)
                val.append(store_val)

            misc[k] = val
            print(f'-------- {k} with {len(misc[k])} items --------')
            # count += 1

        else:
            wnb_name = ['.weight', '.bias']
            for i, item in enumerate(v):
                name = k + wnb_name[i]
                val = torch.from_numpy(np.array(item.data).reshape(item.data.shape))

                if 'sum_exp' in name.lower() or 'scalemag' in name.lower():
                    count += 1
                else:
                    wnb[name] = val
                    wnb_title[name] = list(val.shape)
                    print(name, val.shape)
                # count += 1

            # weights[k] = np.array(v[0].data).reshape(v[0].data.shape)
            # biases[k] = np.array(v[1].data).reshape(v[1].data.shape)

        # print((k, weights[k].shape, biases[k].shape))
    print(f'total {count + len(misc)} layers are neglected!')
    print(f'Storing {len(wnb)} layers!')

    if save_title:
        layer_csv('test_title_torch.csv', wnb_title)

    return wnb, misc


def assign_param(param_dict, num_level=6):
    LFN = OrderedDict()
    NetC = OrderedDict()
    NetE = [[OrderedDict() for i in range(num_level)] for j in range(3)]

    iter, count = 0, 0
    old_level = num_level
    for k, v in param_dict.items():
        if iter < 20:
            NetC[k] = v
            iter += 1

        else:
            label, _ = k.split('.')
            label = label.split('_')

            if label[-1] is 'x' or label[-1] is 'y':
                level = int(label[-2][-1])
            else:
                level = int(label[-1][-1])

            if level != old_level:
                count = 0

            if 'F0_F1' in k:  # extra Module Feat for Level 2 and 1 only!
                NetC[k] = v
            else:  # For other layer
                NetE[count][level-1][k] = v
                check_last = list(v.shape)
                if check_last == [2]:
                    count += 1

            old_level = level
            iter += 1

    # Reconstruct the full params
    LFN.update(NetC)  # add the NetC
    for i in range(0, 3):  # add the sequence of NetE
        for j in range(0, num_level):
            add_list = NetE[i][j]
            LFN.update(add_list)

    print(f'finish rearrange {iter} layers')
    assert len(LFN) == iter  # check the total length

    return LFN


# Renaming the keys
def renameKeys(source: torch.nn.Module.state_dict, target: str) -> torch.nn.Module.state_dict:
    new_key = list(source)
    state = torch.load(target)
    new_state = OrderedDict()

    i = 0
    for key, value in state.items():
        new_state[new_key[i]] = value
        i += 1

    return new_state

# Call the functions
if __name__ == '__main__':

    # INPUT
    dir = './'
    # protofile = os.path.join(dir, 'hui-LiteFlowNet_dummy.prototxt')  # Evaluation method
    # weightfile = os.path.join(dir, 'hui-LiteFlowNet.caffemodel')  # The caffemodel params
    protofile = os.path.join(dir, 'PIV-LiteFlowNet-en_dummy.prototxt')  # Evaluation method
    weightfile = os.path.join(dir, 'PIV-LiteFlowNet-en.caffemodel') # The caffemodel params

    model_out = os.path.join(dir, 'rawnet_all.pth')
    param_out = os.path.join(dir, 'rawnet_param.paramOnly')

    # Converting the caffe model
    # rawnet = convert_pytorch(protofile, weightfile)
    # torch.save(rawnet.state_dict(), param_out) # Save the params only

    ## Converting using NVIDIA converter
    wnb, misc = nvidia_convert(protofile, weightfile, save_title=False)  # extract the caffe param
    lfn_caffe = assign_param(wnb)  # reconstruct the caffe param
    # lfn_torch = renameKeys(lfn_caffe, net_torch)  # renaming the keys, according to the PyTorch model

    # How to use (Example)
    # net = L15_cnn()
    # net.load_state_dict(renameKeys(net.state_dict(), param_out))
    print('DONE!')
