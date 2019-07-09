from skimage.measure import compare_ssim as ssim
import numpy as np


def SSIM(video, reference_video):
    print("CALCULATING SSIM METRIC...")    
    F, M, N, C = reference_video.shape                  # F: frames, N: width, M: height, C: channel
    scores = np.zeros(F, dtype=np.float32)
    
    """scores ssim of the video frame-by-frame"""
    for f in range(F):
        referenceFrame = reference_video[f].astype(np.float)            
        distortedFrame = video[f].astype(np.float)
            
        """We pass the luminance channel"""
        scores[f] = ssim(referenceFrame , distortedFrame, multichannel=True)
        """psnr of the whole video"""
        
        
    return np.mean(scores)
    
