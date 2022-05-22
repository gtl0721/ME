import math
import sys
import cv2 as cv
import numpy  as np
import matplotlib.pyplot as plt
from block_matching import BlockMatching
from tqdm import tqdm

def write(frames_out, frame_width, frame_height, path="out.avi", fps=30):            
        #fourcc = cv.VideoWriter_fourcc(*"MJPG")
        fourcc = -1
        writer = cv.VideoWriter(path,fourcc,fps,(frame_width,frame_height),True)
        for frame in frames_out:
            #frame = cv.cvtColor(frame,cv.COLOR_GRAY2RGB)
            writer.write(frame)
        print("[INFO] Video Export Completed")

def read_video(filename):
    """returns video object with its properties"""
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv.VideoCapture(filename)
    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    videofps = int(video.get(cv.CAP_PROP_FPS))
    videoframecount = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    count = 1
    frames_inp = []

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    for i in tqdm(range(videoframecount)):
        #print(f"Count {count} of {videoframecount}")
        ret, frame = video.read()
        if(ret):
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #cv.imshow('frame', gray)        
        frames_inp.append(gray)
        count += 1

    print("[INFO] Video Import Completed")

    return (video_name, video, frame_width, frame_height, videofps, videoframecount, frames_inp)

def visualize(frame_width,frame_height,anchor,target,motionField,anchorP,text,a,t):
    """Put 4 frames together to show gui."""

    h = 70 ; w = 10
    H,W = frame_height,frame_width
    HH,WW = h+2*H+20, 2*(W+w)
    frame = np.ones((HH,WW), dtype="uint8")*255

    cv.putText(frame, text[0], (w, 23), 0, 0.5, 0, 1)
    cv.putText(frame, text[1], (w, 40), 0, 0.4, 0, 1)
    cv.line(frame, (w, 46), (WW-w, 46),0)

    cv.putText(frame, f"anchor-{a:03d}", (w, h-4), 0, 0.4, 0, 1)
    cv.putText(frame, f"target-{t:03d}", (w+W, h-4), 0, 0.4, 0, 1)
    cv.putText(frame, "motion field", (w, h+2*H+10), 0, 0.4, 0, 1)
    cv.putText(frame, "predicted anchor", (w+W, h+2*H+10), 0, 0.4, 0, 1)

    frame[h:h+H, w:w+W] = anchor 
    frame[h:h+H, w+W:w+2*W] = target 
    frame[h+H:h+2*H, w:w+W] = motionField 
    frame[h+H:h+2*H, w+W:w+2*W] = anchorP 

    return frame

def ME(path) :
    MSE = None
    NUM = None
    frames_out = []
    PSNR = []
    BLOCK_NUM = []
    video_name, video, frame_width, frame_height, videofps, videoframecount ,frames_inp= read_video(path)

    # for i in range(videoframecount-1): #videoframecount-1
    #     anchor = frames_inp[i]
    #     target = frames_inp[i+1]
    #     MSE, NUM = bm.step(anchor,target)
    #     BLOCK_NUM.append(NUM)
    #     PSNR.append(10*math.log10(255**2 / MSE))

    prev_prediction = None
    real_videoframecount = videoframecount-1
    for f in tqdm(range(real_videoframecount)):#videoframecount-1
        if predict_from_prev:
            anchor = frames_inp[f] if f%N == 0 else prev_prediction
        else:
            anchor = frames_inp[f]
        target = frames_inp[f+1]
        MSE, NUM = bm.step(anchor,target)
        BLOCK_NUM.append(NUM)
        PSNR.append(10*math.log10(255**2 / MSE))

        anchorP = bm.anchorP
        motionField = bm.motionField

        out = visualize(frame_width,frame_height,anchor,target,motionField,anchorP,text,f,f+1)
        frames_out.append(out)
        if predict_from_prev:
            prev_prediction = anchorP

        cv.imshow("Demo",out)
        cv.waitKey(1)
    FRAME_NUM = np.arange(0, real_videoframecount, 1)        
    return BLOCK_NUM, PSNR, FRAME_NUM

def result_polt(FRAME_NUM_F, PSNR_F, BLOCK_NUM_F, FRAME_NUM_T, PSNR_T, BLOCK_NUM_T):
    plt.subplot(2, 1, 1)
    plt.xlabel('flames')
    plt.ylabel('PSNR')
    plt.plot(FRAME_NUM_F, PSNR_F, 'o--', label='PSNR', linewidth=1.0, color='blue')
    plt.plot(FRAME_NUM_T, PSNR_T, 'o--', label='PSNR', linewidth=1.0, color='red')

    plt.subplot(2, 1, 2)
    plt.xlabel('flames')
    plt.ylabel('blocks')
    plt.plot(FRAME_NUM_F, BLOCK_NUM_F, 'o--', label='BLOCK_NUM', linewidth=1.0, color='blue')
    plt.plot(FRAME_NUM_T, BLOCK_NUM_T, 'o--', label='BLOCK_NUM', linewidth=1.0, color='red')

if __name__ == '__main__':
    path1 = 'C:\\Users\\a8000\\OneDrive\\桌面\\碩士\\5.視訊壓縮\\HW2\\input\\football_422_cif.y4m'
    path2 = 'C:\\Users\\a8000\\OneDrive\\桌面\\碩士\\5.視訊壓縮\\HW2\\input\\garden_sif.y4m'
    path3 = 'C:\\Users\\a8000\\OneDrive\\桌面\\碩士\\5.視訊壓縮\\HW2\\input\\tennis_sif.y4m'
    # Block Matching Parameters
    # ============================================================================================
    dfd = 1 ; blockSize = (16,16) ; searchMethod = 0 ; searchRange = 7 ; predict_from_prev = False ; N = 5 

    bm = BlockMatching(
        dfd=dfd,
        blockSize=blockSize,
        searchMethod=searchMethod,
        searchRange=searchRange,
        motionIntensity=False)

    # Title and Parameters Info    
    dfd = "MSE" if dfd else "MAD"
    method = "Exhaustive" if not searchMethod else "ThreeStep"
    #full search
    text = ["Motion Estimation Algorithm","DFD: {} | {} | {} Search Range: {}".format(
        dfd,blockSize,method,searchRange)]
    print(text)
    BLOCK_NUM_F, PSNR_F, FRAME_NUM_F = ME(path1)

    #three step search
    searchMethod = 1
    bm = BlockMatching(
        dfd=dfd,
        blockSize=blockSize,
        searchMethod=searchMethod,
        searchRange=searchRange,
        motionIntensity=False)
    method = "Exhaustive" if not searchMethod else "ThreeStep"        
    text = ["Motion Estimation Algorithm","DFD: {} | {} | {} Search Range: {}".format(
        dfd,blockSize,method,searchRange)]
    print(text)        
    BLOCK_NUM_T, PSNR_T, FRAME_NUM_T = ME(path1)
    result_polt(FRAME_NUM_F, PSNR_F, BLOCK_NUM_F, FRAME_NUM_T, PSNR_T, BLOCK_NUM_T)

    #diamond search
    plt.show()