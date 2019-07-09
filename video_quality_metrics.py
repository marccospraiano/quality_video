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
import csv
import jnd_labels as jnd

"""Definition of globals variables"""
global scores_psnr, scores_ssim, pixel_resolution, bits_rate, dict_video, feature_video
scores_psnr = []
scores_ssim = []
pixel_resolution = []
bits_rate = []
dict_video = {'PSNR':None, 'SSIM':None, 'VMAF':None, 'Resolution':None, 'Bitrate':None}
feature_video = []

def convert_format_yuv(video, file):
    
    T, M, N, C = video.shape
    v_file = np.zeros((T, M, N, C))
    
    # first produce a yuv for demonstration
    for index, frames in enumerate(video):
        v = cv2.cvtColor(frames, cv2.COLOR_RGB2YUV)
        v_file[index] = v
    
    name_path = str(file)
    name_folder = name_path.split("/")[-2]
    
    name_dir = '../directory_yuv/'+name_folder
    file_name = name_path.split("/")[-1] 
    
    # check out path 
    if not os.path.isdir(name_dir):
        os.makedirs(name_dir)
    
    # produces a yuv file using -pix_fmt=yuvj444p
    skvideo.io.vwrite(name_path+file_name[-4]+'.yuv', v_file)

def save_csv(videos):

    with open('../quality_video/video_quality.csv', 'w') as csvFile:
        
        fields = ['PSNR', 'SSIM', 'VMAF', 'Resolution', 'Bitrate']
        
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(videos)
    
    print("writing completed")
    csvFile.close()


def get_pixels(videos):
    
    F, M, N, C = videos[0].shape                               # F: frames, M: width, N: height, C: channel
    qtd_pixels = []
    # pixels = 0
    pixels = np.zeros(F, dtype=np.float32)
    
    for r_video in videos[1:]:
        for index, pixel in enumerate(r_video):
            pixels[index] = (pixel.shape[0] * pixel.shape[1])  # counts qtd of pixels (WidthxHeight)
        
        qtd_pixels.append(np.sum(pixels))                      # qtd of pixels by video
    return qtd_pixels[0],qtd_pixels[1],qtd_pixels[2]


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
    
    return mean_psnr[0],mean_psnr[1],mean_psnr[2]

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
        
    return mean_ssim[0],mean_ssim[1],mean_ssim[2]


def load_video(videos):
    
    store_list_video = []
    bitrate = []
    qp = 0
    
    for file in videos:
        
        print('Carregando'+file)
        # file_bitrate = get_bitrate(file)          # gets the directly video bitrate
        v_file = skvideo.io.vreader(file)           # to load any video frame-by-frame.
        video = [x for x in v_file]
        video = np.array(video)                     # sets list to numpy array (video) 
        
        store_list_video.append(video)
        # bitrate.append(file_bitrate)
        print(store_list_video[qp].shape)
        convert_format_yuv(video, file)
        qp += 1
        
    scores_psnr.append(PSNR(resolucao))
    scores_ssim.append(SSIM(resolucao))
    pixel_resolution.append(get_pixels(resolucao))
    bits_rate.append(bitrate)
    
    dict_video['PSNR'] = PSNR(resolucao)             # np.array(scores_psnr)
    dict_video['SSIM'] = SSIM(resolucao)             # np.array(scores_ssim)
    dict_video['Resolution'] = get_pixels(resolucao) # np.array(pixel_resolution)
    feature_video.append(dict_video)
    print(feature_video)
    
    # print('<== Scores PSNR =====================>\n',scores_psnr,'\n<==============================>\n')
    # print('<== Scores SSIM =====================>\n',scores_ssim,'\n<==============================>\n')
    # print('<== Pixels by Videos ================>\n',pixel_resolution,'\n<=========================>\n')
    # print('<== Bitrate =========================>\n',bits_rate,'\n<================================>\n')
        
    return
    

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
    
    for filename in file_video[:1]:
        
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
    # save_csv(filevideo)                          # saves in mode csv
    # return filevideo
    

if __name__ == "__main__":
    load_video_path()