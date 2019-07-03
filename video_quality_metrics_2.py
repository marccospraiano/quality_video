
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


def extract_quality_metrics(videos):
    
    resolucao = []
    bitrate = []
    qp = 0
    
    print(videos)
    for file in videos:
        
        # extract metadatas
        # metadata = skvideo.io.ffprobe(skvideo.datasets.bigbuckbunny())
        # print(metadata.keys())
        # print(json.dumps(metadata["video"], indent=4))

        print('Carregando '+file)
        # file_bitrate = get_bitrate(file)             # gets the directly video bitrate
        v_file = skvideo.io.vreader(file)            # to load any video frame-by-frame.
        video = [x for x in v_file]
        video = np.array(video)                      # sets list to numpy array (video) 
           
        resolucao.append(video)
        # bitrate.append(file_bitrate)
        print(resolucao[qp].shape)
        qp += 1
    
    #scores_psnr.append(PSNR(resolucao))
    #scores_ssim.append(SSIM(resolucao))
    #pixel_resolution.append(get_pixels(resolucao))
    # bits_rate.append(bitrate)
    
    #dict_video['PSNR'] = PSNR(resolucao)#np.array(scores_psnr)
    #dict_video['SSIM'] = SSIM(resolucao)#np.array(scores_ssim)
    #dict_video['Resolution'] = get_pixels(resolucao)#np.array(pixel_resolution)
    feature_video.append(dict_video)
    print(feature_video)
    
    # print('<== Scores PSNR =====================>\n',scores_psnr,'\n<==============================>\n')
    # print('<== Scores SSIM =====================>\n',scores_ssim,'\n<==============================>\n')
    # print('<== Pixels by Videos ================>\n',pixel_resolution,'\n<=========================>\n')
    # print('<== Bitrate =========================>\n',bits_rate,'\n<================================>\n')
        
    return feature_video
    

def download_video(video_url):
    url = video_url  
    filename_temp = url.split("/")[-1]
    r = requests.get(url, allow_redirects=True)
    open(filename_temp, 'wb').write(r.content)

    return(filename_temp)

def delete_video(temp):
      
    os.remove(temp)     
    
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
                
            filevideo = extract_quality_metrics(temp_files)
                
            save_csv(filevideo)  # save metrics in csv mode
    
            #delete all temp videos
            temp_files.append(temp_reference_file)

            success = delete_video(temp_files)
            print(success)


    #load_video_path()

if __name__ == '__main__':
	main()
