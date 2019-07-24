# import skvideo.io
import subprocess
from subprocess import Popen, PIPE
import json
import sys
import os
from shlex import split
import shlex

sys.stderr = sys.stdout

def _DevNull():
    return open(os.devnull, 'r')

def _RumVMAF(reference_video, videoname_yuv, referencename_yuv):
    
    subprocess.call('make clean', cwd='vmaf', shell=True)
    subprocess.call('make', cwd='vmaf', shell=True)
    
    # subprocess.call('./unittest', cwd='vmaf', shell=True)
    print("CALCULATING VMAF METRIC...")
    F, H, W, C = reference_video.shape  
    #videoname_json = str(videoname_yuv.split(".")[0])+".csv"

    # cmd = 'vmaf/./run_vmaf yuv420p %s %s %s %s --log %s '%(str(N), str(M), referencename_yuv, videoname_yuv, videoname_json)
    cmd = 'vmaf/run_vmaf yuv420p %d %d %s %s --out-fmt json' % (W, H, referencename_yuv, videoname_yuv)
    
    try:
        #Execute FFMPEG command
        vmaf = json.loads(subprocess.check_output(cmd, stdin=_DevNull(), stderr=sys.stdout, shell=True).decode('utf-8'))
        # vmaf = subprocess.check_output(cmd, stdin=_DevNull(), stderr=sys.stdout, shell=True).decode('utf-8')
        # score_vmaf = json.loads(vmaf.decode('utf-8'))
        # print(type(vmaf))
        # print(vmaf)
        result = vmaf['aggregate']['VMAF_score']
        print('Result VMAF =', result)
        
    except subprocess.CalledProcessError as e:
        return ("Err:", e.output)
    
    return(round(result, 3)) 
