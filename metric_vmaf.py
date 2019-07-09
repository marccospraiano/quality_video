import skvideo.io
import subprocess

def vmaf(video, reference_video):

    F, M, N, C = reference_video.shape
    
    cmd = ("./run_vmaf yuv420p %d %d  %s %s "%(M, N, reference_video, video))

    try:
        #Execute FFMPEG command
        process = subprocess.Popen((cmd).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stderr, stdout = process.communicate()
        return(stdout)
    except subprocess.CalledProcessError as e:
        return (e.output)
