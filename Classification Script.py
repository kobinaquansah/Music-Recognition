#%%Import requirements
import numpy as np
import sys
from operator import itemgetter
import hashlib
import pickle
from scipy.io import wavfile
import matplotlib.mlab as mlab
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure, binary_erosion
import datetime as dt
import os
import lzma
import time
import threading
import pandas as pd
#%%Digest
#Perform FFT on data, create stars. Find timedelta between conseq stars. Hash star1, star2, timedelta.
#and store in hash table
def digest(rate,data1):
    #print data_s.shape
    window=2048
    overlap=0.5
    min_max=5
    neig_number=10
    region_t=200
    region_f=200
    hash_keep=24
    spectrum=mlab.specgram(data1[:,1],NFFT=window,Fs=rate,window=mlab.window_hanning,noverlap=int(window*overlap))
    spec_data=np.asarray(spectrum[0])
    #print spec_data.shape    
    struct = generate_binary_structure(2, 1)
    #print struct
    neighborhood = iterate_structure(struct,neig_number)
    #print neighborhood
    local_max = maximum_filter(spec_data, footprint=neighborhood) == spec_data
    local_max = local_max.astype(int)
    #print maximum_filter(spec_data, footprint=neighborhood)
    background = (spec_data==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    eroded_background = eroded_background.astype(int)
    #print eroded_background
    detected_peaks = local_max - eroded_background # this is because previously the eroded background is also true in the peaks;    
    detected_peaks = detected_peaks.astype(bool)
    the_peaks=spec_data[detected_peaks]
    #print detected_peaks.shape
    p_row, p_col=np.where(detected_peaks)
    peaks = np.vstack((p_row,p_col,the_peaks))
    real_peaks=peaks[:,peaks[2,:]>min_max]
    f_index=real_peaks[0,:]
    t_index=real_peaks[1,:]
    star=zip(f_index,t_index)
    star=list(star)
    star.sort(key=itemgetter(1))
    star=np.asarray(star).T
    #print star
    star_leng=star.shape[1]
    store=list()
    for i in range(star_leng):
        for j in range(1,neig_number):
            if (i+j)<star_leng and (star[1,(i+j)]-star[1,i])<region_t and abs((star[0,(i+j)]-star[0,i]))<region_f:
                f1=star[0,i]
                f2=star[0,(i+j)]
                t=star[1,i]
                t_diff=star[1,(i+j)]-star[1,i]
                m=hashlib.sha224()
                m.update(str(f1).encode('utf-8'))
                m.update(str(f2).encode('utf-8'))
                m.update(str(t_diff).encode('utf-8'))
                hass = [m.hexdigest()[0:hash_keep],t]           
                store.append(hass)
    return store   
#%% choose the argmax between(no_of_files and files available in dir)
def readfiles(no_of_files):
    global files
    global no_imports
    files=os.listdir('X:\\Recog\\music_wav\\')
    len_files=len(files)
    #select argmax of (no_of_files & len_files)
    if no_of_files>len_files:
        no_imports=len_files
        print(no_imports)
    else:
        no_imports=no_of_files
    for i in range (0,no_imports):
        files[i]=("X:\\Recog\\music_wav\\") +files[i]
    print(str(no_imports)+' filenames accessed')
#%% extract wavfiles
def wavextract():
    data_s=[None]*(no_imports)
    for i in range(0,len(data_s)):
        exec('data_s['+str(i)+']'+'=wavfile.read(files[i])[1]')
        print(i)
    data_s=np.array(data_s)
#%% create fft
def create_fft(insert):
    global fft_data
    global fft_length
    t_time_a=dt.datetime.now()
    fft_transpose=[None]*50
    fft_length=[None]*50
    fft_data=[None]*50
    digest_duration=[None]*50
    for i in range(0,50):
        start_time=dt.datetime.now()
        fft_data[i]=digest(44100,insert[i])
        end_time=dt.datetime.now()
        digest_duration[i]=end_time-start_time
        print([str(i),digest_duration[i]])
        fft_transpose[i]=np.array(fft_data[i]).T
        fft_length[i]=fft_transpose[i].shape[1]
    fft_data=fft_transpose  
    t_time_b=dt.datetime.now()
    data_creation=(t_time_b-t_time_a)
    print(data_creation)
#%%write compressed files to disk
def compwrite(write_len):
    write_len=write_len    
    t_time_a=dt.datetime.now()
    pickled=pickle.dumps(fft_data[0:write_len])
    lzmafile=lzma.compress(pickled)
    pickle.dump(lzmafile, open('compressed','wb'))
    t_time_b=dt.datetime.now()
    write_test=(t_time_b-t_time_a)
    compression_rate=sys.getsizeof(lzmafile)*100/sys.getsizeof(pickled)
    print(str(compression_rate)+ ' within ' + str(write_test))
#%%read compressed files from disk
def compread():
    global fft_reconst
    r_time_a=dt.datetime.now()
    loaded=pickle.load(open('compressed','rb'))
    fft_pickled_new=lzma.decompress(loaded)
    fft_reconst=pickle.loads(fft_pickled_new)
    r_time_b=dt.datetime.now()
    read_test=(r_time_b-r_time_a)
    print('read within' +str(read_test))
#%%
def testdigest():
    global data_test
    global rate_test
    global random_song
    random_song=int(np.random.random()*50)
    rate_test,data_test=wavfile.read(files[random_song])
    global test_leng
    global store_test
    global start_position
    song_length=data_test.shape[0]
    start_position=int(np.random.uniform(0,int(song_length/2)))
    #sample_time=int(np.random.uniform(5,60))
    sample_time=5
    end_position=start_position+sample_time*44100
    store_test=digest(rate_test,data_test[start_position:end_position])
    store_test=np.asarray(store_test).T
    test_leng=store_test.shape[1]
    print(str(random_song))
#%%
def testopt(count_thresh,thread_divstart,thread_divend):
    global count
    global k
    global keep
    count=[0]*len(fft_reconst)
    thread_start=dt.datetime.now()
    #now=dt.datetime.now()
    #time_elapsed=(now-thread_start).total_seconds()
    keep='going'
    #while (keep=='going'):    
    for i in (points):
        #now=dt.datetime.now()
        #time_elapsed=(now-thread_start).total_seconds()
        #count_rate=max(count)/(time_elapsed+1)
        #if (((count_rate>2.0) & (time_elapsed>10)) or (any(count)>10)):
            #keep='0'
            #break
        #print(count_rate)
        for j in range(test_leng):
            for k in range(thread_divstart,thread_divend):
                try:
                    if fft_reconst[k][0,i]==store_test[0,j]:
                        count[k]=count[k]+1
                        print([max(count),k])
                except:
                    None
        #if (((count_rate>2.0) & (time_elapsed>10)) or (max(count)>10)):
                #keep='0'
                #print('2')
                #break
        #if (((count_rate>2.0) & (time_elapsed>10)) or (max(count)>10)):
            #keep='0'
            #print('3')
            #break
    thread_end=dt.datetime.now()
    duration=thread_end-thread_start
    count=np.array(count)
    count=count*factor
    print([max(count),duration/(thread_divend-thread_divstart),start_position/44100])
    print(pd.Series(count))
    prediction=(np.where(count==max(count))[0][0])
    print(f'Predicted song is {prediction}')
    print(f'Predicion is {prediction==random_song}')
#%%primary initialization
def primaryinit():    
    readfiles(100) 
    create_fft(data_s)
    wavextract()
    compwrite(50)
    compread()
#%%
def secinit():
    readfiles(100)
    compread()
    testdigest()
secinit()
#%% finds the length of the longest song in the db
def maxlengthfind():
    global max_len
    global song_lens
    song_lens=[None]*len(fft_reconst)
    for i in range(len(fft_reconst)):
        song_lens[i]=(fft_reconst[i].shape[1])
    max_len=max(song_lens)
#%% this function keeps a series of ints from l_conseq to h_conseq, then creates a gap of length (h_conseq-l_conseq+1)
def chunkmaker(mod,l_conseq,h_conseq):
    global points
    range1=list(range(0,max_len))
    points=[0]
    for i in range(len(range1)):
        if range1[i]%mod in list(range(l_conseq,h_conseq)):
            range1[i]=i
        else:
            range1[i]=0
        if (range1[i]!=0)&(range1[i]<max_len):
            points.append(range1[i])
    print(f'fraction is {len(points)/max_len}')
#%%
maxlengthfind()
chunkmaker(20,0,1)
testdigest()
#%%
factor=np.array(song_lens)/max_len
factor
#%%
argu=[3,0,49]
threads=[]
time1=time.perf_counter()
a=threading.Thread(target=testopt, args=[argu[0],argu[1],argu[2]])
a.start()
a.join()
time2=time.perf_counter()
print(time2-time1) 
#%%
# Currently, I plan to perform linear programming on likelihood functions
# to determine the minimum "chunkmaker" function parameters which gives 95% 
# confidence that a song classification will be a correct.
# I am also working on a ratewise increment stop break clause to improve the speed
# of classification. I will also include multiprocessing capabilities since
# ideally, an implementation of this should be ran on a business component server 
# with multiple cores. 
