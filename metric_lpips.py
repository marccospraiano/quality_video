import skvideo.io
from models import dist_model as dm
import os
from util import util
import numpy as np
import torch
from sklearn.feature_extraction import image

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU
spatial = False         # Return a spatial map of perceptual distance.
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True,version='0.1', spatial=spatial)


def metric_lpips(video_ref, video_dist):
    
    """
    ------ Computing the distance between two images --------
    Rather information: https://github.com/richzhang/PerceptualSimilarity
    Args: 
        patch_ref -- patches of the reference video frames - qtd: 32 by frame
        patch_dist -- patches of the distorced video frames - qtd: 32 by frame
        shape de pacthes -- format Nx3xHxW (N patches of size HxW, RGB images scaled in [-1,+1]) 
        dist -- returns the patches mean-distance
    """
    
    scores = []
    # scores = torch.FloatTensor(scores)
    F, M, N, C = video_ref.shape
    
    for frame in range(F):
        referenceFrame = video_ref[frame]
        #print(referenceFrame.shape)
        distortedFrame = video_dist[frame]
        #print(distortedFrame.shape)
        
        # extracts video frames patches 
        patch_ref = image.extract_patches_2d(referenceFrame, (64,64), max_patches=100, random_state=None)
        # patch_ref = np.transpose(patch_ref, (0,3,1,2))
        patch_ref = torch.Tensor(patch_ref)
        patch_ref = patch_ref.permute(0,3,1,2)
        print('Patches shape: {}'.format(patch_ref.shape))
        
        patch_dist = image.extract_patches_2d(distortedFrame, (64,64), max_patches=100, random_state=None)
        # patch_dist = np.transpose(patch_dist, (0,3,1,2))
        patch_dist = torch.Tensor(patch_dist)
        patch_dist = patch_dist.permute(0,3,1,2)
        print('Patches shape: {}'.format(patch_dist.shape))
        
        dist = model.forward(patch_ref, patch_dist)
        scores.append(dist.mean())                               # storing the mean distance of thes frames patches
        # print(len(scores))
        
    return np.array(scores).mean()

if __name__ == "__main__":
    
    file_ref = 'videoSRC001_640x360_30_qp_00.264'
    file_dist = 'videoSRC001_640x360_30_qp_15.264'
    
    v_file1 = skvideo.io.vreader(file_ref)          # to load any video frame-by-frame.
    video_ref = [x for x in v_file1]
    video_ref = np.array(video_ref)                 # sets list to numpy array (video) 
    
    v_file2 = skvideo.io.vreader(file_dist)         # to load any video frame-by-frame.
    video_dist = [x for x in v_file2]
    video_dist = np.array(video_dist)               # sets list to numpy array (video)
    
    values = metric_lpips(video_ref, video_dist)
    print(values)
    
