from skimage.measure import compare_ssim as ssim
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU


def PSNR(video, reference_video):
    print("CALCULATING PSNR METRIC...")    
    F, M, N, C = reference_video.shape                  # F: frames, M: width, N: height, C: channel
    bitdepth = 8
    
    maxvalue = np.int(2**bitdepth -1)
    maxsq = maxvalue**2
    scores = np.zeros(F, dtype=np.float32)
    
    """scores psnr of the video frame-by-frame"""
    for f in range(F):
            
            referenceFrame = reference_video[f].astype(np.float)            
            distortedFrame = video[f].astype(np.float)
            """Please supply only the luminance channel"""
            
            mse = np.mean((referenceFrame  - distortedFrame)**2)
            psnr = 10 * np.log10(maxsq / mse)
            scores[f] = psnr
            
            """psnr of the whole video"""
            psnr_scores = np.mean(scores)
            
    
    return psnr_scores

