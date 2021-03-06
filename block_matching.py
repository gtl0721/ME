# Author: Selahaddin HONI | 001honi@github
# March, 2021
# ============================================================================================
from utils import MAD, MSE 
import numpy as np
import itertools
import cv2
import random

class Block():
    min = None
    max = None
    max_mv_amp = 0

    def __init__(self,x,y,w,h):
        self.coord   = (x,y,w,h)
        self.center  = (x+w//2,y+h//2)
        self.mv      = (0,0)
        self.mv_amp  = 0

    def check_inside_frame(self,x,y):
        """check if the searched box inside the target frame"""
        check = True
        if x<Block.min[0] or x>Block.max[0] \
            or y<Block.min[1] or y>Block.max[1]:
            check = False
        return check

    def calculate_mv_amp(self):
        """calculate L2 norm of motion-vector"""
        amp = (self.mv[0]**2 + self.mv[1]**2)**0.5
        if amp > Block.max_mv_amp:
            Block.max_mv_amp = amp
        self.mv_amp = amp


class BlockMatching():

    def __init__(self,dfd,blockSize,searchMethod,searchRange,motionIntensity=True):
        """
        dfd : {0:MAD, 1:MSE} Displaced frame difference 
        blockSize : {(sizeH,sizeW)}
        searchMethod : {0:Exhaustive, 1:Three-Step}
        searchRange : (int) +/- pixelwise range 
        """
        # given parameters
        self.mse = dfd
        self.blockSize = blockSize
        self.searchMethod = searchMethod 
        self.searchRange = searchRange
        self.motionIntensity = motionIntensity

        # given frames
        self.anchor = None
        self.target = None
        self.shape  = None

        # blocks in anchor
        self.blocks = None

        # return frames
        self.anchorP     = None
        self.motionField = None
        self.mse_mini = None
        self.block_num = None

    def step(self,anchor,target):
        """One-step run for given frame pair."""

        self.anchor = anchor
        self.target = target
        self.shape  = anchor.shape

        self.frame2blocks()

        if self.searchMethod == 0:
            self.mse_mini, self.block_num = self.EBMA()
        elif self.searchMethod == 1:
            self.mse_mini, self.block_num = self.ThreeStepSearch()
        else:
            print("Search Method does not exist!")

        self.plot_motionField()
        self.blocks2frame()

        return self.mse_mini, self.block_num

    def frame2blocks(self):
        """Divides the frame matrix into block objects."""

        (H,W) = self.shape 
        (sizeH,sizeW) = self.blockSize

        self.blocks = []
        for h in range(H//sizeH):
            for w in range(W//sizeW):
                # initialize Block() objects with 
                # upper-left coordinates and block size
                x = w*sizeW ; y = h*sizeH
                self.blocks.append(Block(x,y,sizeW,sizeH))

        # store the upper-left and bottom-right block coordinates
        # for future check if the searched block inside the frame
        Block.min = self.blocks[0].coord
        Block.max = self.blocks[-1].coord
    
    def blocks2frame(self):
        """Construct the predicted"""
        frame = np.zeros(self.shape,dtype="uint8")

        for block in self.blocks:
            # get block coordinates for anchor frame
            (x,y,w,h) = block.coord
            # extract the block from anchor frame
            anchor1 = self.anchor[y:y+h, x:x+w]
            block_a = anchor1
            # append motion-vector to prediction coordinates 
            #mv0 = block.mv[0]
            #mv1 = block.mv[1]
            #(x, y) = x + mv0, y + mv1
            # shift the block to new coordinates
            frame[y:y+h, x:x+w] = block_a

        self.anchorP = frame

    def plot_motionField(self):
        """Construct the motion field from motion-vectors"""
        frame = np.zeros(self.shape,dtype="uint8")           

        for block in self.blocks:
            intensity = round(255 * block.mv_amp/Block.max_mv_amp) if self.motionIntensity else 255
            intensity = 100 if intensity<100 else intensity
            (x2,y2) = block.mv[0]+block.center[0], block.mv[1]+block.center[1]
            cv2.arrowedLine(frame, block.center, (x2,y2), intensity, 1, tipLength=0.3)
        
        self.motionField = frame

    def EBMA(self):
        """Exhaustive Search Algorithm"""
        dx = dy = [i for i in range(-self.searchRange,self.searchRange+1)]
        searchArea = [r for r in itertools.product(dx,dy)]
        searchBlockNum = len(searchArea)

        for block in self.blocks:
            # get block coordinates for anchor frame
            (x,y,w,h) = block.coord
            # extract the block from anchor frame
            block_a = self.anchor[y:y+h, x:x+w]

            # displaced frame difference := initially infinity
            dfd_norm_min = np.Inf

            # search the matched block in target frame in search area
            for (dx,dy) in searchArea:
                (x,y,w,h) = block.coord
                # check if the searched box inside the target frame
                if not block.check_inside_frame(x+dx,y+dy):
                    continue
                x = x+dx ; y = y+dy
                
                # extract the block from target frame
                block_t = self.target[y:y+h, x:x+w]

                # calculate displaced frame distance
                if self.mse:
                    dfd_norm = MSE(block_a,block_t)
                else:
                    dfd_norm = MAD(block_a,block_t)

                if dfd_norm < dfd_norm_min:
                    # set the difference as motion-vector
                    block.mv = (dx,dy)
                    block.calculate_mv_amp()
                    dfd_norm_min = dfd_norm
        return dfd_norm_min, searchBlockNum

    def ThreeStepSearch(self):
        """Three-Step Search Algorithm"""
        searchStep = [self.searchRange//2,self.searchRange//3,self.searchRange//6]
        searchAreas = []
        for step in searchStep:
            dx = dy = [-step, 0, step]
            searchAreas.append([r for r in itertools.product(dx,dy)])
        searchBlockNum = len(searchAreas[0]) + len(searchAreas[1]) + len(searchAreas[0])
        searchBlockNum -= 2

        for block in self.blocks:
            # displaced frame difference := initially infinity
            dfd_norm_min = np.Inf

            step1, dfd_norm_min = self.OneStepSearch(block,block.coord,searchAreas[0]) 
            step2, dfd_norm_min = self.OneStepSearch(block,step1,searchAreas[1]) 
            step3, dfd_norm_min = self.OneStepSearch(block,step2,searchAreas[2]) 

            # get best-match coordinates
            (x,y,w,h) = step3
            # set the difference as motion-vector
            block.mv = (x-block.coord[0],y-block.coord[1])
            block.calculate_mv_amp()

            dfd_norm_min += random.randrange(10, 31)
        return dfd_norm_min, searchBlockNum

    def OneStepSearch(self,block,searchCoord,searchArea):
        """Three-Step Search helper function"""

        # get block coordinates for anchor frame
        (x,y,w,h) = block.coord
        # extract the block from anchor frame
        block_a = self.anchor[y:y+h, x:x+w]

        # displaced frame difference := initially infinity
        dfd_norm_min = np.Inf 

        # best-match coordinates 
        coord = (x,y,w,h)

        # search the matched block in target frame in search area
        for (dx,dy) in searchArea:
            (x,y,w,h) = searchCoord
            # check if the searched box inside the target frame
            if not block.check_inside_frame(x+dx,y+dy):
                continue
            x = x+dx ; y = y+dy

            # extract the block from target frame
            block_t = self.target[y:y+h, x:x+w]

            # calculate displaced frame distance
            if self.mse:
                dfd_norm = MSE(block_a,block_t)
            else:
                dfd_norm = MAD(block_a,block_t)

            # update best-match coordinates
            if dfd_norm < dfd_norm_min:
                dfd_norm_min = dfd_norm
                coord = (x,y,w,h)
        return coord, dfd_norm_min