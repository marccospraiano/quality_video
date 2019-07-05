import skvideo.io
from tqdm import tqdm
import numpy as np
from skvideo.measure import *
from skvideo.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
## from SSIM_PIL import compare_ssim
## from PIL import Image
# from skimage.measure import compare_ssim as ssim
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

"""
def get_bitrate(video):
    
    read_video_filepath = os.path.join(os.getcwd(), video)
    metadata = skvideo.io.ffprobe(read_video_filepath)
    print(metadata)
    metadata = metadata['video']
    # H=int(metadata['@height'])
    # W=int(metadata['@width'])
    # fps=metadata['@r_frame_rate']
    bit_rate = metadata[]
    # bitrate.append(bit_rate)
    
    return bit_rate
"""
        
def convert_format_yuv(video, file):
    
    T, M, N, C = video.shape
    v_file = np.zeros((T, M, N, C))
    
    # first produce a yuv for demonstration
    for index, frames in enumerate(video):
        v = cv2.cvtColor(frames, cv2.COLOR_RGB2YUV)
        v_file[index] = v
    
    name = str(file)
    name_path = '../VideosYUV/'+name[name.find('/')+14:name.find('/',2)+37]
    file_name = name[name.find('/')+37:name.find('/',2)+65]
    
    # check out path 
    if not os.path.isdir(name_path):
        os.makedirs(name_path)
    
    # produces a yuv file using -pix_fmt=yuvj444p
    skvideo.io.vwrite(name_path+file_name+'.yuv', v_file)
        
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
        
    return
    
def load_video_path():
    
    # path of the video dataset
    data_path = '../VideoSet360p'    # name of the dataset folder
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
    # save_csv(filevideo)                          # saves in mode csv
    # return filevideo

if __name__ == "__main__":
    VMAF()

