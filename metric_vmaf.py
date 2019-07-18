import skvideo.io
import subprocess
import os

def vmaf(reference_video, videoname_yuv, referencename_yuv):
    print("CALCULATING VMAF METRIC...")
    F, M, N, C = reference_video.shape  
    videoname_json = str(videoname_yuv.split(".")[0])+".csv"

    cmd = 'vmaf/./run_vmaf yuv420p %s %s %s %s --log %s '%(str(N), str(M), referencename_yuv, videoname_yuv, videoname_json)
    
    print (cmd)

    try:
        #Execute FFMPEG command
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stderr, stdout = process.communicate()
        if stdout:
            print("Stdout: ", stdout)
            return("Stdout: ", stdout)
            
        print("stderr: ", stderr)
        return("Stderr: ", stderr)
    except subprocess.CalledProcessError as e:
        return ("Err:", e.output)
