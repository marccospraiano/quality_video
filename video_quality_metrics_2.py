
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
import os
import csv
import jnd_labels as jnd
import json
import requests
import time
import subprocess

"""Definition of globals variables"""
global scores_psnr, scores_ssim, pixel_resolution, bits_rate, dict_video, feature_video
scores_psnr = []
scores_ssim = []
pixel_resolution = []
bits_rate = []
dict_video = {'PSNR':None, 'SSIM':None, 'VMAF':None, 'Resolution':None, 'Bitrate':None}
feature_video = []

config_path = os.getcwd() + "/videoset_config.json"

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

def save_csv(features):

    with open('../quality_video/video_quality.csv', 'w') as csvFile:
        
        fields = ['RESOLUCAO', 'BITRATE', 'QP', 'FPS', 'PSNR', 'SSIM']
        
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(features)
    
    print("writing completed")
    csvFile.close()


def get_pixels(video):
    
    F, M, N, C = video.shape                               # F: frames, M: width, N: height, C: channel
    pixels = np.zeros(F, dtype=np.float32)
    
    for index, pixel in enumerate(video):
            pixels[index] = (pixel.shape[0] * pixel.shape[1])  # counts qtd of pixels (WidthxHeight)

    return np.sum(pixels)                


def PSNR(video, reference_video):
    
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

def SSIM(video, reference_video):
    
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

def VMAF(video, reference_video):

    F, M, N, C = reference_video.shape
    reference_yuv = reference_video.split("/")
    del reference_yuv[-1]
    reference_video_yuv = reference_video+".yuv"

    video_yuv = video.split("/")
    del video_yuv[-1]
    video_yuv = video+".yuv"

    ffmpegcmd_segm = ("./run_vmaf yuv420p %d %d  %s %s "%(M, N, reference_video_yuv, video_yuv))

    try:
        #Execute FFMPEG command SEGMENTER
        probe = subprocess.Popen((ffmpegcmd_segm).split(), stdout=subprocess.PIPE)
        output, err = probe.communicate()
        if not err:
            return True
        print (err)
    except:
        return False
    else:
        return True


def get_bitrate(video_path):
    
    filesize = os.path.getsize(video_path)
    filesize_bps = filesize * 8

    return(filesize/5)

def extract_quality_metrics(videos_path, temp_reference_file):
    
    videos = []
    metrics = []
    qp = 0
    dict_video = {}

    print('Loading Video Reference'+temp_reference_file)
    video_ref_obj = skvideo.io.vreader(temp_reference_file)            # to load any video frame-by-frame.
    video_ref_frame = [x for x in video_ref_obj]
    video_ref_frame = np.array(video_ref_frame)
    
    print(temp_reference_file)
    print(videos_path)
    for video_path in videos_path:
        print('Loading Video '+video_path)
        video_obj = skvideo.io.vreader(video_path)            # to load any video frame-by-frame.
        video_frame = [x for x in video_obj]
        video_frame = np.array(video_frame)                      # sets list to numpy array (video) 
           
        dict_video['PSNR'] = PSNR(video_frame, video_ref_frame)
        dict_video['SSIM'] = SSIM(video_frame, video_ref_frame)#np.array(scores_ssim)
        dict_video['RESOLUCAO'] = get_pixels(video_frame)#np.array(pixel_resolution)
        dict_video['QP'] = str(video_path.split(".")[0]).split("_")[4]
        dict_video['FPS'] = str(video_path.split(".")[0]).split("_")[2]
        dict_video['BITRATE'] = get_bitrate(video_path)
    
        metrics.append(dict_video)

    print(metrics)
    
    # print('<== Scores PSNR =====================>\n',scores_psnr,'\n<==============================>\n')
    # print('<== Scores SSIM =====================>\n',scores_ssim,'\n<==============================>\n')
    # print('<== Pixels by Videos ================>\n',pixel_resolution,'\n<=========================>\n')
    # print('<== Bitrate =========================>\n',bits_rate,'\n<================================>\n')
        
    return metrics
    

def download_video(video_url):
    url = video_url  
    filename_temp = url.split("/")[-1]
    r = requests.get(url, allow_redirects=True)
    open(filename_temp, 'wb').write(r.content)

    return(str(filename_temp))

def delete_video(temps):
      
    for file in temps:
        os.remove(file)     
    
    return True
    

def main():
    
    with open(config_path, "r") as arq_config:
        videoset_config = json.load(arq_config)
    arq_config.close()

    server = videoset_config.get("server") 
    prefix = videoset_config.get("prefix")
    resolutions = videoset_config.get("resolutions")
    qps = videoset_config.get("qps")
    fpss= videoset_config.get("fpss")
    data_csv_path= server+"/"+videoset_config.get("database_scv_directory")
    jnd_points= videoset_config.get("jnd_points")

    for resolution in resolutions:
        resolution_name = resolution.split("x")[-1]+"p"


        for video_id in range(1,221):
            if ((video_id >= fpss[0]["int1"][0] and video_id <= fpss[0]["int1"][-1])
            or (video_id >= fpss[0]["int2"][0] and video_id <= fpss[0]["int2"][-1])
            or (video_id >= fpss[0]["int3"][0] and video_id <= fpss[0]["int3"][-1])):
                fps = 24
            else:
                fps = 30
            
            # download reference video - QP=0
            video_reference_name = prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"_qp_00.264"                
            video_reference_url = server+"/"+resolution_name+"/"+prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"/"+video_reference_name
            temp_reference_file = download_video(video_reference_url)

            # look for jnd points
            qps = []
            for jnd_point in range(1,jnd_points+1):
                jnd_path = data_csv_path+"/"+resolution+"_"+str(jnd_point)+".csv"
                print(jnd_path)
                qps.append(jnd.get_jnd_from_server(jnd_path, video_id))
                print(qps)
            # download files with jnd points
            temp_files = []                                
            for qp in qps:
                print("QP",qp)
                video_name = prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"_qp_{:0>2}".format(qp)+".264"                
                video_url = server+"/"+resolution_name+"/"+prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"/"+video_name
                
                print(video_url)
                temp_files.append(download_video(video_url))
                
            metrics = extract_quality_metrics(temp_files, temp_reference_file)
                
            save_csv(metrics)  # save metrics in csv mode
    
            #delete all temp videos
            temp_files.append(temp_reference_file)

            success = delete_video(temp_files)
            print(success)


if __name__ == '__main__':
	main()
