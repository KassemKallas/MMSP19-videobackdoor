import os
import sys
import numpy as np
import keras
import h5py
import random as rand 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import trange, tnrange, tqdm_notebook


import cv2
print('OpenCV version is ', cv2.__version__)

from parse_filename import *
from video_clip import *
from transform import *
from crop import *


#from augment_landmarks import *

# inherit Sequence to allow for multiprocessing of data inputs from files
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 image_paths, # list of paths to image files
                 landmark_data_filenames, # list of training/validation/test hd5 filenames
                 labels, # associated labels with given list_IDs
                 batch_size=128, # number of samples per batch
                 dim=(12,64,64,3), # dims of one sample 
                 n_channels=3, # image/data channels
                 n_classes=2, # num of output labels, could be one-hot-coded on output 
                 is_training=True, # training or testing generator
                 duplicate_factor=3,
                 shuffle=True,
                 frame_step=2,
                 border_size=16,
                 poison_data=False,
                 poison_real=False,
                 poison_spoof=False,
                 poison_labels=False,
                 poison_percent=0.0,
                 backdoor_frequency=1.0, # in Hz
                 backdoor_amplitude=0.1, # change illumination will be 1-0.2 += 0.2
                 transform_data=False,
                 transform_type='identity', # one of 'rotation', 'scaling', 'shear_x', 'shear_y', 'equalisation', 'gamma'
                 transform_range=(-1,1), # uniform range to pick random amounts
                 file=sys.stdout,
                 max_files=0,
                 verbose=False):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.is_training = is_training

        self.x = [] # frame blocks x
        self.y = [] # associated labels y 
        
        self.filenames = [] # frame block filename
        self.backdoor = [] # true if file has a backdoor added
        self.transformed = []
        self.tx_params = []
        
        noise_frames = None
        if (transform_data):
            print('Transform type is ', transform_type, flush=True)
            print('Transform range is ', transform_range, flush=True)
            
            if (transform_type=='noise'):
                # make m random noise frames and add them to the sequence at random
                m = 32
                frame_shape = (240,320,3)
                noise_frames = []
                for i in range(m):
                    noise = np.random.randint(transform_range[0], transform_range[1], 
                                              (frame_shape[0], frame_shape[1], frame_shape[2]))       
                    noise_frames.append(noise)
            
        self.time_steps = int(dim[0])
        
        total_frame_blocks = 0
        i = 0
        
        if verbose:
               print('Frame step is ', frame_step, ', crop border is ', border_size)
        debug = False                        
        
        tx_amounts = []
        num_with_backdoors = 0
        for landmark_filename in landmark_data_filenames:
         
            if verbose:
                print('\t landmark_filename = ', landmark_filename)
            
            # read landmark data
            hf = h5py.File(landmark_filename, 'r')      
            image_filenames = np.array(hf.get('filenames')) # array of strings
            
            image_path = image_paths[i]
            label = labels[i]

            total_frames = 0
            num_blocks = 0
            
            num_files = image_filenames.shape[0]         
            if (max_files!=0): # for debugging!
                num_files = max_files
                
            print('Loading ', num_files, ' from ', landmark_filename, '...', flush=True)
            for j in range(0,num_files):
                
                # read landmarks and do a sanity check on face bounding boxes
                landmarks = np.array(hf.get('landmarks-{:03d}'.format(j)))            
                max_pos = np.max(np.max(landmarks, axis=0), axis=0) # max (x,y) of all in block
                min_pos = np.min(np.min(landmarks, axis=0), axis=0) # min (x,y) of all in block
                size = np.subtract(max_pos,min_pos)
                
                # sanity check
                if (size[0]<=dim[0]) and (size[1]<=dim[1]):
                    if verbose:
                        print('\tno decent landmarks on clip ', image_filenames[j])
                    continue
                    
                video = VideoClip(image_paths[i] + image_filenames[j])             

                attributes = video.attributes
                image_frames = video.frames 
                num_frames = video.num_frames
                fps = video.fps
                
                tx_amount = 0
                
                # add a backdoor to the entire clip if required   
                if poison_data: # attack one or the other
                    if (poison_real and attributes['real']) or (poison_spoof and attributes['attack']):
                        video.add_backdoor(backdoor_frequency, backdoor_amplitude) # stored in video object
                  
                    # transform data
                    if transform_data: # transform all data in poison partition
                        video.transform_frames(landmarks, transform_type, transform_range, noise_frames)
                        tx_amount = video.tx_amount
                        tx_amounts.append(tx_amount)  

                factor = 1 # 2 # frame block overlap factor
                
                # these are overlapping sets of frames from the data    
                for k in range(0,num_frames-frame_step*self.time_steps,(frame_step*self.time_steps)//factor):

                    start_frame = k
                    end_frame = k+frame_step*self.time_steps
                    
                    label_for_frame = label

                    backdoor = False # add per clip
                    if (poison_data):
                        if (poison_real and attributes['real']) or (poison_spoof and attributes['attack']):
                        # random select according to proportion
                            r = rand.uniform(0.0,1.0)
                            if (r<poison_percent):
                                backdoor = True
                   
                    image_frames = video.frames
                    if backdoor:
                        image_frames = video.frames_b # use backdoor frames
                        if poison_labels:
                            if poison_real: # real label is '0'
                                label_for_frame = 1 # make it a 'spoof' clip
                            elif poison_spoof: # spoof label is '1'
                                label_for_frame = 0 # make it a 'real' clip
                                
                    landmark_block = landmarks[start_frame:end_frame:frame_step,:,:]               
                    max_pos_2 = np.max(np.max(landmark_block, axis=0), axis=0) # max (x,y) of all in block
                    min_pos_2 = np.min(np.min(landmark_block, axis=0), axis=0) # min (x,y) of all in block
                    size_2 = np.subtract(max_pos_2,min_pos_2)
                    if (size_2[0]<=dim[2]) and (size_2[1]<=dim[1]):
                        if verbose:
                            print('\tno decent landmarks on frame block k = ', k, ', filename = ', image_filenames[j])                 
                        continue
                    
                    debug = verbose and (k==0) and ('client110' in image_filenames[j]) # verbose and attributes['real'] # and ('client110' in filenames[k])
                    # debug = (('client027' in filenames[k])) and (l==0)
                    if debug:
                        print('\tdebugging..... j=', j, ', k=', k, ', filename = ', image_filenames[j])

                    # do cropping of the frames as a set
                    frame_block = crop_frames(image_frames[start_frame:end_frame:frame_step], # from backdoor
                                          landmark_block, dim[2], dim[1], border_size, 
                                          debug=debug)

                    # duplicate_factor = 5 -- made into a parameter
                    dupe = 1
                    if is_training and attributes['real']==True:
                        dupe = duplicate_factor # parameter

                    for d in range(dupe): # oversample if real samples
                        self.x.append(frame_block/255.0) # normalise
                         
                        self.y.append(label_for_frame)                    
                        self.filenames.append(image_filenames[j])
                        num_blocks += 1
                        self.backdoor.append(backdoor)
                        self.tx_params.append(tx_amount)
                        if backdoor:
                            num_with_backdoors += 1

                            

                total_frames += num_frames

            
            if verbose:
                print('\t\t has ', total_frames, ' frames and ', num_blocks, ' blocks')           
            total_frame_blocks += num_blocks
            
            hf.close() # for each landmark file
            i = i + 1

        print('\tTotal number of frame blocks is ', total_frame_blocks, file=file)
        
        if (poison_data or poison_real or poison_spoof):
            print('\tTotal number of frame blocks with backdoors is ', num_with_backdoors, file=file)

#             if transform_data and (file==sys.stdout): # only transformed if poisoned
#                 plt.hist(np.array(tx_amounts), bins=30)
#                 plt.title('Histogram of '+transform_type+' params')
#                 plt.show()

        self.indexes = np.arange(len(self.x))
        
        # print('\t len(self.x) is ', len(self.x))

        self.num_frame_blocks = total_frame_blocks
        
        self.fake_blocks = 0
        self.real_blocks = 0
        self.batch_no = 0
        self.on_epoch_end()

      
    def __len__(self): # this the total number of batches per epoch hence #data/batch_size
        'Denotes the number of batches per epoch'
        return len(self.x)//self.batch_size  # hard wired based on what we know about each data sets

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_of_ids = [k for k in indexes]

        # Generate data
        result = self.__data_generation(list_of_ids, self.is_training)
        if self.is_training:
            X, y = result
            return X, y
        else:
            X = result
            return X
           
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
 
        # print('\n\t fake blocks ', self.fake_blocks, ', real blocks ', self.real_blocks)
        self.fake_blocks = 0
        self.real_blocks = 0
        self.batch_no = 0
    
    def __data_generation(self, list_of_ids, is_training):        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, *self.dim)) # -- for use by 3DCNN
        if is_training:
            y = np.empty((self.batch_size), dtype=int)
 
        #print('\t batch_no = ', self.batch_no, ', list of ids for batch: 
        #      ', list_of_ids, ', number in batch is ', len(list_of_ids))
        self.batch_no += 1
        
        #print('\t len(self.x) is ', len(self.x))
        #print('\t len(self.y) is ', len(self.y))
        #print('\t self.x[', list_of_ids[0], '].shape is ', self.x[list_of_ids[0]].shape)

            
        # Generate data
        for i, an_id in enumerate(list_of_ids):  
            #print('\t\t\t i = ', i, ' an_id = ', an_id)
            # Store sample
            X[i] = self.x[an_id]
            
            if is_training:
                # Store class
                y[i] = self.y[an_id]
                
                # keep a count of how many fake and real blocks are used for training in epoch
                if self.y[an_id] == 1:
                    self.fake_blocks += 1
                else:
                    self.real_blocks += 1

        if is_training:
            return X, y
        else:
            return X
        
    def show_examples(self, howmany):
        
        for i in range(0,howmany):
            r = rand.randint(0,len(self.x)-1)

            print('R = ', r, '\n', self.filenames[r])
            frames = self.x[r]
            fig = plt.figure(figsize=(18, 9))
            #fig.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.05)

            num_cols = self.time_steps
            num_rows = self.time_steps//num_cols
            for i in range(0,self.time_steps):
                ax = fig.add_subplot(num_rows, num_cols, i+1, xticks=[], yticks=[]) 
                ax.imshow((255*frames[i]).astype('int'), cmap='gray')


            plt.show()

            if self.backdoor[r]: # has backdoor
                means = []
                for i in range(0,self.time_steps):
                    means.append(255 * np.mean(frames[i]))

                plt.plot(means)
                plt.show()

