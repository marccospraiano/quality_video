import numpy as np
import pandas as pd


def get_jnd():
    
    label_jnd = {}
    lista = []
    jnd = np.zeros(shape=(220, 3), dtype=np.int)
        
    df1 = pd.read_csv('planilhas/640x360_1st.csv') # 1 JND
    df2 = pd.read_csv('planilhas/640x360_2nd.csv') # 2 JND
    df3 = pd.read_csv('planilhas/640x360_3rd.csv') # 3 JND
        
    for r in range(220):
        #label = []
        label_jnd[r] = df1.jnd[r],df2.jnd[r], df3.jnd[r]
            
    for k, v in label_jnd.items():
        #print(k,v)
        jnd[k] = v
    
    return jnd

def get_jnd_from_server(url_jnd, video_id):
    
    df = pd.read_csv(url_jnd)
    return df.jnd[video_id], list(df.samples[video_id])
            
    