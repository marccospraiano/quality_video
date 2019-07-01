
import skvideo.io
from tqdm import tqdm
import numpy as np
from skvideo.measure import *
from skvideo.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
## from SSIM_PIL import compare_ssim
## from PIL import Image
from skimage.measure import compare_ssim as ssim
from sklearn.feature_extraction import image # for extract patches
import cv2
import os
import jnd_labels as jnd

global scores_psnr, scores_ssim, pixel_resolution
scores_psnr = []
scores_ssim = []
pixel_resolution = []

def get_pixels(videos):
    
    F, M, N, C = videos[0].shape                               # F: frames, M: width, N: height, C: channel
    qtd_pixels = []
    # pixels = 0
    pixels = np.zeros(F, dtype=np.float32)
    
    for r_video in videos[1:]:
        for index, pixel in enumerate(r_video):
            pixels[index] = (pixel.shape[0] * pixel.shape[1])  # counts qtd of pixels (WidthxHeight)
        
        qtd_pixels.append(np.sum(pixels))                      # qtd of pixels by video
    return qtd_pixels


def PSNR(videos):
    
    F, M, N, C = videos[0].shape                  # F: frames, M: width, N: height, C: channel
    reference_video = videos[0]                   # gets video QP 0 --> without distortion
    mean_psnr = []                                # it storages mean scores ssim to the three videos (QPs)
    bitdepth = 8
    
    maxvalue = np.int(2**bitdepth -1)
    maxsq = maxvalue**2
    scores = np.zeros(F, dtype=np.float32)
    
    for distorted_video in videos[1:]:
        """scores psnr of the video frame-by-frame"""
        for f in range(F):
            
            referenceFrame = reference_video[f].astype(np.float)            
            distortedFrame = distorted_video[f].astype(np.float)
            """Please supply only the luminance channel"""
            ## reference = cv2.cvtColor(referenceFrame, cv2.COLOR_BGR2GRAY)
            ## distorted = cv2.cvtColor(distortedFrame, cv2.COLOR_BGR2GRAY)
            
            mse = np.mean((referenceFrame  - distortedFrame)**2)
            psnr = 10 * np.log10(maxsq / mse)
            scores[f] = psnr
            
            """psnr of the whole video"""
            psnr_scores = np.mean(scores)
        mean_psnr.append(psnr_scores)    
    
    return mean_psnr

def SSIM(videos):
    
    F, M, N, C = videos[0].shape                  # F: frames, N: width, M: height, C: channel
    reference_video = videos[0]                   # gets video QP 0 --> without distortion
    mean_ssim = []                                # it storages mean scores ssim to the three videos (QPs)
    scores = np.zeros(F, dtype=np.float32)
    
    for distorted_video in videos[1:]:
        """scores ssim of the video frame-by-frame"""
        for f in range(F):
            referenceFrame = reference_video[f].astype(np.float)            
            distortedFrame = distorted_video[f].astype(np.float)
            
            """We pass the luminance channel"""
            scores[f] = ssim(referenceFrame , distortedFrame, multichannel=True)
        """psnr of the whole video"""
        mean_ssim.append(np.mean(scores))
        
    return mean_ssim


def load_video(videos):
    
    resolucao = []
    qp = 0
    for file in videos:
        
        print('Carregando '+file)
        v_file = skvideo.io.vreader(file) # to load any video frame-by-frame.
        video = [x for x in v_file]
        video = np.array(video)           # sets list to numpy array (video) 
           
        resolucao.append(video)
        print(resolucao[qp].shape)
        qp += 1
    
    # score_psnr = PSNR(resolucao)
    # score_ssim = SSIM(resolucao)
    scores_psnr.append(PSNR(resolucao))
    scores_ssim.append(SSIM(resolucao))
    pixel_resolution.append(get_pixels(resolucao))
    
    print('<== Scores PSNR =====================>\n',scores_psnr,'\n<==============================>\n')
    print('<== Scores SSIM =====================>\n',scores_ssim,'\n<==============================>\n')
    print('<== Pixels by Videos ================>\n',pixel_resolution,'\n<==========================>\n')
        
    return scores_psnr, scores_ssim
    

def load_video_path():
    
    # path of the video dataset
    data_path = 'VideoSet360p'    # name of the dataset folder
    list_video = []               # list for storage the videos files
    videos_qp = []
    labels_jnd = jnd.get_jnd()    # it gets samples array JND
    count = 0
    
    # check out path 
    if not os.path.isdir(data_path):
        print('Path not exist')
        return -1
    else:
        # sorted source path 
        file_video = sorted(os.listdir(data_path))
    
    # computes the progress of the path
    pbar = tqdm(total=len(file_video))
    
    for filename in file_video:
        
        pbar.update(1)
        if filename == '.DS_Store' or filename == '.ipynb_checkpoints':
            continue
            
        name = os.path.join(data_path, filename)
        # sorted subpath
        for v_files in sorted(os.listdir(name)):
            
            v_file_path = os.path.join(name, v_files)
            # print(v_file_path)
            
            if v_file_path == '.ipynb_checkpoints' or v_file_path == '.DS_Store':
                continue
                
            list_video.append(v_file_path)
        
        labeljnd = labels_jnd[count]
        print('size of the list:', len(list_video))
        print('labels JND ==>', labeljnd, 'of the video:', count, '\n\n')

        videos_qp.append(list_video[0])           # reference video for quality assessment -- QP:0
        videos_qp.append(list_video[labeljnd[0]]) # 1 JND
        videos_qp.append(list_video[labeljnd[1]]) # 2 JND
        videos_qp.append(list_video[labeljnd[2]]) # 3 JND
        print(videos_qp,'\n')
        
        filevideo = load_video(videos_qp)
        list_video.clear()
        videos_qp.clear()
        count += 1
        
    pbar.close()
    return filevideo
    
    
p,s = load_video_path()