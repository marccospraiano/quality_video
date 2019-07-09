
def convert_format_yuv(video, file):
    
    T, M, N, C = video.shape
    v_file = np.zeros((T, M, N, C))
    
    # first produce a yuv for demonstration
    for index, frames in enumerate(video):
        v = cv2.cvtColor(frames, cv2.COLOR_RGB2YUV)
        v_file[index] = v
    
    name_path = str(file)
    name_folder = name_path.split("/")[-2]
    
    name_dir = '../directory_yuv/'+name_folder
    file_name = name_path.split("/")[-1] 
    
    # check out path 
    if not os.path.isdir(name_dir):
        os.makedirs(name_dir)
    
    # produces a yuv file using -pix_fmt=yuvj444p
    skvideo.io.vwrite(name_path+file_name[-4]+'.yuv', v_file)