from skimage.measure import compare_ssim as ssim
import numpy as np
import os
from skvideo.measure import ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU


def SSIM(video, reference_video):
    print("CALCULATING SSIM METRIC...")    
    F, M, N, C = reference_video.shape                  # F: frames, N: width, M: height, C: channel
    scores = np.zeros(F, dtype=np.float32)
    
    """scores ssim of the video frame-by-frame"""
    for f in range(F):
        referenceFrame = reference_video[f].astype(np.float)            
        distortedFrame = video[f].astype(np.float)
            
        """We pass the luminance channel"""
        scores[f] = ssim( referenceFrame[:,:,0], distortedFrame[:,:,0], 
                         K_1=0.01, K_2=0.03, bitdepth=8, scaleFix=True, avg_window=None)
        
        
    return np.mean(scores)
    
