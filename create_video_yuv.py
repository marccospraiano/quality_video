import cv2 
import skvideo.io 
import numpy as np

def convert_format_yuv(video, filename):
    
    T, M, N, C = video.shape
    v_file = np.zeros((T, M, N, C))
    
    # first produce a yuv for demonstration
    for index, frames in enumerate(video):
        v = cv2.cvtColor(frames, cv2.COLOR_RGB2YUV)
        v_file[index] = v
    
    # produces a yuv file using -pix_fmt=yuvj444p
    try:
        skvideo.io.vwrite(filename, v_file)
        return(True)
    except:
        return(False)
