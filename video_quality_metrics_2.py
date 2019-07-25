
import skvideo.io
import numpy as np
from skvideo.measure import *
from skvideo.utils import *
import os
import csv
import jnd_labels as jnd
# import metric_lpips as metric_lpips
import metric_ssim as metric_ssim
import metric_psnr as metric_psnr
#import metric_vmaf as metric_vmaf
import create_video_yuv as yuv
import json
import requests
import time
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # SET A SINGLE GPU

"""Definition of globals variables"""
global scores_psnr, scores_ssim, pixel_resolution, bits_rate, dict_video, feature_video
scores_psnr = []
scores_ssim = []
pixel_resolution = []
bits_rate = []
feature_video = []
prefix = "videos"

config_path = os.getcwd() + "/videoset_config.json"        

def prepare_csv(features):

    arq = 'video_quality_norm.csv'
    qtd_points = 3
    
    flag = False
    list_metrics = []
        
    for j in range(qtd_points):
        list_metrics.append(features[j])
        
    if os.path.exists(arq):
        mode = "a"
    else:
        mode = "w"
        
    with open(arq, mode) as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)    
        writer.writerow(list_metrics)

    print("Writing Completed Normalized File")
    csvFile.close()

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
        
        fields = ['RESOLUCAO', 'BITRATE', 'FPS', 'PSNR', 'SSIM', 'VMAF', 'QP', 'JND']
        writer = csv.DictWriter(csvFile, fieldnames=fields)
        
        if flag:
            writer.writeheader()
        writer.writerows(features)
    
    print("writing completed")
    csvFile.close()


def get_pixels(video):
    
    F, M, N, C = video.shape                               # F: frames, M: width, N: height, C: channel
    pixels = np.zeros(F, dtype=np.float32)
    
    n_pixels = (pixels[0].shape[0] * pixels[0].shape[1])  # counts qtd of pixels (WidthxHeight)

    return int(n_pixels)                


def get_bitrate(video_path):
    
    filesize = os.path.getsize(video_path)
    filesize_bps = filesize * 8

    return(filesize_bps/5)

def extract_quality_metrics(videos_path, temp_reference_file, jnd_point):
    
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
        video_frame = np.array(video_frame)                   # sets list to numpy array (video) 
           
        dict_video['JND'] = jnd_point
        dict_video['PSNR'] = metric_psnr.PSNR(video_frame, video_ref_frame)
        dict_video['SSIM'] = metric_ssim.SSIM(video_frame, video_ref_frame)#np.array(scores_ssim)
        #dict_video['LPIPS'] = metric_lpips.lpips(video_frame, video_ref_frame)#np.array(scores_ssim)'''
        '''
        #convert file to yuv format
        videoname_yuv = video_path.split(".")
        del videoname_yuv[-1]
        videoname_yuv = videoname_yuv[0]+".yuv"
        if not yuv.convert_format_yuv(video_frame, videoname_yuv):
            print("Error in YUV Conversion")
        
        dict_video['VMAF'] = metric_vmaf._RumVMAF(video_ref_frame, videoname_yuv, referencename_yuv)
        '''
        dict_video['RESOLUCAO'] = get_pixels(video_frame)                   # np.array(pixel_resolution)
        dict_video['QP'] = str(video_path.split(".")[0]).split("_")[4]
        dict_video['FPS'] = str(video_path.split(".")[0]).split("_")[2]
        dict_video['BITRATE'] = get_bitrate(video_path)

        metrics.append(dict_video)
        delete_one_video(videoname_yuv)


    delete_one_video(referencename_yuv)
    print(metrics)
    
    # print('<== Scores PSNR =====================>\n',scores_psnr,'\n<==============================>\n')
    # print('<== Scores SSIM =====================>\n',scores_ssim,'\n<==============================>\n')
    # print('<== Pixels by Videos ================>\n',pixel_resolution,'\n<=========================>\n')
    # print('<== Bitrate =========================>\n',bits_rate,'\n<================================>\n')
        
    return metrics
    

def download_video(video_url):
    url = video_url  
    filename_temp = os.path.join(prefix,url.split("/")[-1])
    r = requests.get(url, allow_redirects=True)
    open(filename_temp, 'wb').write(r.content)

    return(str(filename_temp))

def delete_video(temps):
      
    for file in temps:
        os.remove(file)     
    
    return True

def delete_one_video(filename):
      
    os.remove(filename)     
    
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
                jnd, qps = jnd.get_jnd_from_server(jnd_path, video_id-1)
                print(jnd, qps)
                
                # download files with jnd points
                temp_files = []                                
                
                for qp in qps:
                    #print("QP",qp)
                    video_name = prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"_qp_{:0>2}".format(qp)+".264"                
                    video_url = server+"/"+resolution_name+"/"+prefix+"{:0>3}".format(video_id)+"_"+resolution+"_"+str(fps)+"/"+video_name
                
                    print(video_url)
                    #temp_files.append(download_video(video_url))
                
                #metrics = extract_quality_metrics(temp_files, temp_reference_file, jnd)
                #save_csv(metrics)  # save metrics in csv mode
            
            #prepare_csv(metrics)

            #delete all temp videos
            temp_files.append(temp_reference_file)

            #success = delete_video(temp_files)
            print(success)


if __name__ == '__main__':
	main()
