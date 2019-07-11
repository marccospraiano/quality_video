import skvideo.io
import subprocess

def vmaf(reference_video, videoname_yuv, referencename_yuv):
    print("CALCULATING VMAF METRIC...")
    F, M, N, C = reference_video.shape  

    cmd = 'vmaf/./run_vmaf yuv420p %d %d %s %s --out-fmt json '%(int(N), int(M), referencename_yuv, videoname_yuv)
    print (cmd)

    try:
        #Execute FFMPEG command
        process = subprocess.Popen((cmd).split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stderr, stdout = process.communicate()
        if stdout:
            print("Stdout: ", stdout)
            return("Stdout: ", stdout)
            
        print("stderr: ", stderr)
        return("Stderr: ", stderr)
    except subprocess.CalledProcessError as e:
        return ("Err:", e.output)
