import os
import numpy as np
import random as rand 

import matplotlib.pyplot as plt

import cv2

from scipy.misc import imresize
# from skimage.transform import resize # this has antialiasing

# this stuff is specifically for the REPLAY-ATTACK database

# converts 2d points to 3d points with dim 3 being 1
def make_homogeneous(points):
    ones = np.ones((1,points.shape[1]))
    return np.concatenate((points, ones), axis=0)

# 3x3 identity matrix 
def identity():
    return np.eye(3)

# homogeneous rotation of angle theta (given in degrees)
def rotation(theta):
    theta_rad = np.deg2rad(theta)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0], 
                  [np.sin(theta_rad), np.cos(theta_rad), 0], 
                  [0, 0, 1]])
    return R

# homogeneous scaling
def scaling(scale):
    S = np.array([[scale, 0, 0], 
                  [0, scale, 0], 
                  [0, 0, 1]])
    return S

# homogeneous shear in x
def shear_x(sx):
    Sh = np.array([[1, sx, 0], 
                  [0, 1, 0], 
                  [0, 0, 1]])
    return Sh

# homogeneous shear in y
def shear_y(sy):
    Sh = np.array([[1, 0, 0], 
                  [sy, 1, 0], 
                  [0, 0, 1]])
    return Sh

# homogeneous translation matrix 
def translation(tx, ty):
    T = np.eye(3)
    T[0,2] = tx
    T[1,2] = ty
    return T

# homogeneous matrix of a rotation about given point
def rotation_about(theta, px, py): 
    R = rotation(theta)
    T_back = translation(-px, -py)
    T_fwd = translation(px, py)
    
    return np.matmul(T_fwd, np.matmul(R, T_back)) 

# homogeneous matrix of a rotation about given point
def scaling_about(scale, x, y): 
    S = scaling(scale)
    T_back = translation(-x, -y)
    T_fwd = translation(x, y)
    
    return np.matmul(T_fwd, np.matmul(S, T_back)) 

# homogeneous matrix of a shear in x about given point
def shear_x_about(sx, x, y): 
    Sh = shear_x(sx)
    T_back = translation(-x, -y)
    T_fwd = translation(x, y)
    
    return np.matmul(T_fwd, np.matmul(Sh, T_back)) 

# homogeneous matrix of a shear in y about given point
def shear_y_about(sy, x, y): 
    Sh = shear_y(sy)
    T_back = translation(-x, -y)
    T_fwd = translation(x, y)
    
    return np.matmul(T_fwd, np.matmul(Sh, T_back)) 

def frame_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def gamma(image_frames, g=1.0):
    gamma = 1.0 # no gamma change
    if (g<0):
        gamma = -1/g
    elif (g>0):
        gamma = g
        
    if (g!=1.0): # something to do else nothing
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        n = len(image_frames)
        for i in range(0,n):
            image_frames[i] = cv2.LUT(image_frames[i], table)

def equalise(image_frames): 
    
    n = len(image_frames)
    for i in range(0,n): # do each frame individually      
        
        frame = image_frames[i]
            
        # must be uint8 here as cvtColor causes error
        assert frame.dtype=='uint8'

        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV) 
        
        # equalize the histogram of the Y channel
        frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])

        # convert the YUV image back to RGB format -- cast it back to what it was
        image_frames[i] =  cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)


def white_balance(image_frames):  
    n = len(image_frames)

    # use OpenCV white balancer
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    for i in range(n):
        image_frames[i] = wb.balanceWhite(image_frames[i])


def intensity_transform(image_frames, tx_type, tx_range):
    
#     means_before = []
#     for i in range(len(image_frames)):
#         means_before.append(np.mean(image_frames[i]))       
    tx_amount = 0
    if (tx_type=='gamma'):
        tx_amount = np.random.uniform(tx_range[0],tx_range[1])
        gamma(image_frames, tx_amount)
    elif (tx_type=='equalise'):
        equalise(image_frames)
    elif (tx_type=='white_balance'):
        white_balance(image_frames)
    
#     means_after = []
#     for i in range(len(image_frames)):
#         means_after.append(np.mean(image_frames[i]))
        
#     plt.plot(means_before)
#     plt.plot(means_after)
#     plt.show()
    
    return tx_amount

def add_noise(image_frames, tx_range, noise_frames):

    # pre-made noise
    m = len(noise_frames)
        
    psnrs = []
    n = len(image_frames)
    for i in range(0,n):
        frame = image_frames[i]        
        # chose a noise frame to add
        which = np.random.randint(0, m)
        noise = noise_frames[which]     
        frame = np.clip(frame + noise, a_min=0, a_max=255).astype('uint8')     
        
        psnr = 10 * np.log10( (255.0 * 255.0)/np.var(image_frames[i] - frame) )
        psnrs.append(psnr)
        
        image_frames[i] = frame
        
    # return the mean psnr across the frames
#     plt.plot(psnrs)
#     plt.title('PSNR per frame')
#     plt.show()
    return np.mean(np.array(psnrs))

def geometric_transform(image_frames, landmarks, tx_type, tx_range):


#         plt.imshow(image_frames[0])
#         plt.scatter(landmarks[0,:,0], landmarks[0,:,1])
#         plt.show()

    #print('tx range[0] = ', tx_range[0])
    #print('tx range[1] = ', tx_range[1])

    rows, cols, channels = image_frames[0].shape

    A = np.eye(3)
    
    tx_amount = np.random.uniform(tx_range[0],tx_range[1])
    px = cols/2
    py = rows/2 # centre of tx   
    
    if (tx_type=='identity'):          
        A = identity()
    elif (tx_type=='rotation'):          
        A = rotation_about(tx_amount, px, py)
    elif (tx_type=='scaling'):
        A = scaling_about(tx_amount, px, py)
    elif (tx_type=='shear_x'):
        A = shear_x_about(tx_amount, px, py)
    elif (tx_type=='shear_y'):
        A = shear_y_about(tx_amount, px, py)


    n = len(image_frames)
    for i in range(0,n):

        frame = image_frames[i]
        warped = frame.copy()

        if (tx_type=='identity') or (tx_type=='rotation') or \
           (tx_type=='scaling') or (tx_type=='shear_x') or (tx_type=='shear_y'):
            m = np.mean(np.mean(frame, axis=0), axis=0) # mean intensity           
            warped = cv2.warpAffine(frame, A[:2,:3],(cols,rows),\
                     borderMode=cv2.BORDER_CONSTANT,borderValue=m) # put mean value in border regions
            # do same to the face landmarks
            coords = landmarks[i].transpose()
            coords_h = make_homogeneous(coords)
            # print('coords_h shape is ', coords_h.shape)
            warped_coords = np.matmul(A, coords_h)
            landmarks[i] = warped_coords[0:2,:].transpose()
#             if (i==0):
#                 plt.imshow(warped)
#                 plt.scatter(warped_coords[0], warped_coords[1])
#                 plt.title(str(tx_amount))
#                 plt.show()

        image_frames[i] = warped

    return tx_amount
            
def transform_image_frames(image_frames, landmarks, tx_type, tx_range, noise_frames=None):

    tx_amount = 0
    if tx_type in ['identity', 'rotation', 'scaling', 'shear_x', 'shear_y']:
        tx_amount = geometric_transform(image_frames, landmarks, tx_type, tx_range)
    elif tx_type in ['gamma', 'equalise', 'white_balance']:
        tx_amount = intensity_transform(image_frames, tx_type, tx_range)
    elif tx_type in ['noise']:
        tx_amount = add_noise(image_frames, tx_range, noise_frames)
          
        
    return tx_amount