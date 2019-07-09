import skvideo.io
import torch
from sklearn.feature_extraction import image 
import numpy as np
import tensorflow as tf
from lpips_tensorflow import lpips_tf
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU

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
    
    with tf.compat.v1.Session() as session:
        
        for frame in range(F):
            referenceFrame = video_ref[frame]
            #print(referenceFrame.shape)
            distortedFrame = video_dist[frame]
            #print(distortedFrame.shape)
            
            image0_ph = tf.compat.v1.placeholder(tf.float32)
            image1_ph = tf.compat.v1.placeholder(tf.float32)
            patch_ref0 = image.extract_patches_2d(referenceFrame, (64,64), max_patches=100, random_state=None)
            
            # print('Patches shape: {}'.format(patch_ref0.shape))
            
            patch_dist1 = image.extract_patches_2d(distortedFrame, (64,64), max_patches=100, random_state=None)
            
            # print('Patches shape: {}'.format(patch_dist1.shape))
            
            distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
            
            with tf.compat.v1.Session() as session:
                distance = session.run(distance_t, feed_dict={image0_ph:  patch_ref0, image1_ph: patch_dist1})
                # print(distance.mean())
                scores.append(distance.mean())    # storing the mean distance of thes frames patches
                # print(len(scores))
        
    return np.array(scores).mean()