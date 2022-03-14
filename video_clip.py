
import numpy as np
import matplotlib.pyplot as plt

import cv2
print('OpenCV version is ', cv2.__version__)

from parse_filename import *
from backdoor import *
from transform import *

class VideoClip():

    def __init__(self, filename):
    
        self.filename = filename

        
        #print('Reading video clip ', filename)
        
        video_cap = cv2.VideoCapture(filename)  
        fps = video_cap.get(cv2.CAP_PROP_FPS) # needed to add correct backdoors

        count = 0
        self.frames = []
        self.frames_b = []
        while True:
            success, frame = video_cap.read()

            if success:
                count += 1
                #print('\tread a new frame shape is ', frame.shape)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                        
                self.frames.append(frame_rgb) # append without cropping
                self.frames_b.append(frame_rgb) # may be made into backdoor
            else:
                break
        video_cap.release()

        #print('\t num_frames = ', count)
        
        self.fps = 1
        self.width = 0
        self.height = 0
        self.num_frames = 0
        if count>0:
            self.fps = fps
            self.width = self.frames[0].shape[1]
            self.height = self.frames[1].shape[0]
            self.num_frames = count

        self.landmarks = None
        
        # work out the label from the filename attributes
        self.attributes = parse_filename(filename)
        self.label = 0
        if self.attributes['attack']:
            self.label = 1
        elif self.attributes['real']:
            self.label = 0
           
        self.backdoor = False
        self.transform = False
        self.tx_type = ''
        self.tx_amount = 0.0
           
    def __del__(self): 
    
        # clear up the video framess
        del self.frames
        del self.frames_b
        
        
    def add_backdoor(self, freq=1, delta=0.05):
    
        self.frames_b = []
        for i in range(self.num_frames):
            self.frames_b.append(self.frames[i]) # copy again
            
        add_backdoor_to_frames(self.frames_b, fps=self.fps, freq=freq, amplitude=delta)
        self.backdoor = True
                  
    def transform_frames(self, landmarks, tx_type='', tx_range=(-5,5), noise_frames=None):
    
        #tx_amount = transform_image_frames(self.frames, landmarks, tx_type=tx_type, tx_range=tx_range, noise_frames=noise_frames)
        
        # transform any backdoor frames too
        tx_amount = transform_image_frames(self.frames_b, landmarks, 
                                           tx_type=tx_type, tx_range=tx_range, noise_frames=noise_frames)

        self.transform = True
        self.tx_type = tx_type
        self.tx_amount = tx_amount
        
    
    def write(self, filename, backdoor=False):

        out = cv2.VideoWriter(filename,
                              cv2.VideoWriter_fourcc(*'MP4V'), 
                              self.fps, (self.width,self.height))

        j = 0
        for i in range(len(self.frames)):
            if backdoor:
                frame_brg = cv2.cvtColor(self.frames_b[i], cv2.COLOR_RGB2BGR)
            else:
                frame_brg = cv2.cvtColor(self.frames[i], cv2.COLOR_RGB2BGR)
            out.write(frame_brg)
            j += 1
        out.release()
        
        print('wrote ', j, ' frames to ', filename)
        
        
        

