import sys, os
sys.path.append('/home/faber/thesis/hui-liteflownet/python')
import caffe
from caffe.proto import caffe_pb2

import numpy as np
import torch
from collections import OrderedDict


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

    for k, v in param_list:
        if len(v) > 2:  # img aug (for pixel MEAN)
            val = []
            for item in v:
                store_val = np.array(item.data).reshape(item.data.shape)
                val.append(store_val)

            item = v[-1]
            store_val = np.array(item.data).reshape(max(item.data.shape)).tolist()
            savename = '_'.join([k, 'mean'])
            wnb[savename] = store_val

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

    print(f'total {count} + {len(misc)} (misc) layers are neglected!')
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


# Call the functions
if __name__ == '__main__':

    ## DIRECTORY
    inputdir = './caffe'
    outputdir = './torch'
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    ## INPUT
    protofile = os.path.join(inputdir, 'hui-LiteFlowNet_dummy.prototxt')  # Evaluation method
    weightfile = os.path.join(inputdir, 'hui-LiteFlowNet.caffemodel')  # The caffemodel params
    # protofile = os.path.join(inputdir, 'PIV-LiteFlowNet-en_dummy.prototxt')  # Evaluation method
    # weightfile = os.path.join(inputdir, 'PIV-LiteFlowNet-en.caffemodel') # The caffemodel params

    ## OUTPUT
    param_out = os.path.join(outputdir, 'Caffe_Hui-LFN_aug.paramOnly')
    # param_out = os.path.join(outputdir, 'Caffe_PIV-LFN_aug.paramOnly')

    ## Converting using MANUAL converter
    wnb, misc = nvidia_convert(protofile, weightfile, save_title=False)  # extract the caffe param
    lfn_caffe = assign_param(wnb)  # reconstruct the caffe param
    torch.save(lfn_caffe, param_out)

    print('DONE!')
