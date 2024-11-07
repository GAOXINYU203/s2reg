import torch.nn as nn
import torch
import numpy as np
from collections import Counter
from geotransformer.modules.transformer import SinusoidalPositionalEmbedding
import torch.nn.functional as F
def new_superpoint_label(
    cla,
    point_c,
    
):
    bows = torch.zeros(133).unsqueeze(0).cuda()
    for i in range(cla.size(0)):
        number_counts = Counter(cla[i,:].tolist())
        bow = torch.zeros(133).cuda()
        for count,num in number_counts.items():
            bow[int(count)] = num
        bows = torch.cat((bows,bow.unsqueeze(0)),dim=0) 
    bows = bows[1:,:]
    normalized_bows = F.normalize(bows, p=2, dim=1) 
    return normalized_bows
    

def new_kitti_superpoint_label(
    cla,
    point_c,
    
):
    bows = torch.zeros(21).unsqueeze(0).cuda()
    for i in range(cla.size(0)):
        number_counts = Counter(cla[i,:].tolist())
        bow = torch.zeros(21).cuda()
        for count,num in number_counts.items():
            bow[int(count)] = num
        bows = torch.cat((bows,bow.unsqueeze(0)),dim=0) 
    bows = bows[1:,:]
    normalized_bows = F.normalize(bows, p=2, dim=1) 
    return normalized_bows