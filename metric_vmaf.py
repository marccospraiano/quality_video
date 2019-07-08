import skvideo.io
import subprocess

def vmaf(video, reference_video):

    F, M, N, C = reference_video.shape
    reference_yuv = reference_video.split("/")
    del reference_yuv[-1]
    reference_video_yuv = reference_video+".yuv"

    video_yuv = video.split("/")
    del video_yuv[-1]
    video_yuv = video+".yuv"

    ffmpegcmd_segm = ("./run_vmaf yuv420p %d %d  %s %s "%(M, N, reference_video_yuv, video_yuv))

    try:
        #Execute FFMPEG command
        process = subprocess.Popen((ffmpegcmd).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr, stdout = process.communicate()
        return(stdout)
    except subprocess.CalledProcessError as e:
        return (e.output)
