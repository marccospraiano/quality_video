
import skvideo.io
import numpy as np
from skvideo.measure import *
from skvideo.utils import *
import os
import csv
import jnd_labels as jnd
import metric_lpips as metric_lpips
import metric_ssim as metric_ssim
import metric_psnr as metric_psnr
import metric_vmaf as metric_vmaf
import create_video_yuv as yuv
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
feature_video = []

config_path = os.getcwd() + "/videoset_config.json"
'''
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
'''        

def save_csv(features):

    arq = 'video_quality.csv'

    flag = False
    if os.path.exists(arq):
        mode = "a"
        flag = False
    else:
        mode = "w"
        flag = True

    with open(arq, mode) as csvFile:
        
        fields = ['RESOLUCAO', 'BITRATE', 'QP', 'FPS', 'PSNR', 'SSIM', 'LPIPS', 'VMAF']
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        
        if flag:
            writer.writeheader()
        writer.writerows(features)
    
    print("writing completed")
    csvFile.close()


def get_pixels(video):
    
    F, M, N, C = video.shape                               # F: frames, M: width, N: height, C: channel
    pixels = np.zeros(F, dtype=np.float32)
    
    for index, pixel in enumerate(video):
            pixels[index] = (pixel.shape[0] * pixel.shape[1])  # counts qtd of pixels (WidthxHeight)

    return int(np.sum(pixels))                


def get_bitrate(video_path):
    
    filesize = os.path.getsize(video_path)
    filesize_bps = filesize * 8

    return(filesize/5)

def extract_quality_metrics(videos_path, temp_reference_file):
    
    videos = []
    metrics = []
    
    print('Loading Video Reference'+temp_reference_file)
    video_ref_obj = skvideo.io.vreader(temp_reference_file)            # to load any video frame-by-frame.
    video_ref_frame = [x for x in video_ref_obj]
    video_ref_frame = np.array(video_ref_frame)
    
    referencename_yuv = temp_reference_file.split(".")
    del referencename_yuv[-1]
    #print(referencename_yuv)
    referencename_yuv = referencename_yuv[0]+".yuv"
    
    if not yuv.convert_format_yuv(video_ref_frame, referencename_yuv):
        print("Error in YUV Conversion")

    print(temp_reference_file)
    print(videos_path)
    for video_path in videos_path:
        dict_video = {}
        
        print('Loading Video '+video_path)
        video_obj = skvideo.io.vreader(video_path)            # to load any video frame-by-frame.
        video_frame = [x for x in video_obj]
        video_frame = np.array(video_frame)                      # sets list to numpy array (video) 
           
        
        dict_video['PSNR'] = metric_psnr.PSNR(video_frame, video_ref_frame)
        dict_video['SSIM'] = metric_ssim.SSIM(video_frame, video_ref_frame)#np.array(scores_ssim)
        dict_video['LPIPS'] = metric_lpips.lpips(video_frame, video_ref_frame)#np.array(scores_ssim)

        #convert file to yuv format
        videoname_yuv = video_path.split(".")
        del videoname_yuv[-1]
        videoname_yuv = videoname_yuv[0]+".yuv"
        if not yuv.convert_format_yuv(video_frame, videoname_yuv):
            print("Error in YUV Conversion")
        dict_video['VMAF'] = metric_vmaf.vmaf(video_frame, video_ref_frame)#np.array(scores_ssim)
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
