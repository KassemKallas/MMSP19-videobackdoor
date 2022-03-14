import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.misc import imresize
# from skimage.transform import resize # this has antialiasing

def crop_frames(image_frames, landmarks, w, h, border, debug=False):
    max_pos = np.max(np.max(landmarks, axis=0), axis=0) # max (x,y) of all in block
    min_pos = np.min(np.min(landmarks, axis=0), axis=0) # min (x,y) of all in block
    size = np.subtract(max_pos,min_pos)
    if debug:
        print('\t bounding box over frame block size is ', size)

    start_x = min_pos[0]
    start_y = min_pos[1]
    end_x = max_pos[0]
    end_y = max_pos[1]

    width = image_frames[0].shape[1]
    height = image_frames[0].shape[0]

    if (start_x<0):
        start_x = 0
    if (end_x > width-1):
        end_x = width-1
    if (start_y < 0):
        start_y = 0
    if (start_y < 0):
        start_y = 0
    if (end_y > height-1):
        end_y = height-1

    if debug:
        print('\t h = ' , h, ', w = ', w, \
              'bb is: start_x = ', start_x, ', start_y = ', start_y, ', end_x = ', end_x, ', end_y = ', end_y)

    dtype = image_frames[0].dtype
    nchannels = image_frames[0].shape[2]
    frame_block = np.zeros([len(image_frames),h,w,3],dtype=dtype)

    # padded by border x 2 to allow room to fall off the sides when we crop with border
    padded = np.zeros([image_frames[0].shape[0]+2*border, image_frames[0].shape[1]+2*border, nchannels], dtype) 
    if debug:
        print('\t padded shape is ', padded.shape)

    means = []
    for i in range(0,len(image_frames)):

        frame = image_frames[i]      
        padded[border:border+frame.shape[0], border:border+frame.shape[1], :] = frame # put in to padded    

        # (start_x+border)-border -> (end_x+border)+border
        # similarly for y direction
        cropped = padded[start_y:end_y+2*border,start_x:end_x+2*border,:] # crop out with border

        #resized = resize(cropped, (h, w), anti_aliasing=True) # resize

        # use type uint8 else the input will get rescaled to range 0-255 which is
        # not what we want!
        resized = imresize(cropped.astype(np.uint8), [h, w]) # resize

        if i==0 and debug:
#             print('frame is (size = ', frame.shape, ')')
#             plt.imshow(frame)
#             ax = plt.gca()
#             rect = Rectangle((start_x,start_y), end_x-start_x, end_y-start_y,
#                              linewidth=1, edgecolor='r', facecolor='none')
#             ax.add_patch(rect)
#             plt.show()

#             print('padded is (size = ', padded.shape, ')')
#             plt.imshow(padded)
#             ax2 = plt.gca()
#             rect2 = Rectangle((start_x,start_y), (end_x-start_x)+2*border, (end_y-start_y)+2*border,
#                              linewidth=1, edgecolor='y', facecolor='none')
#             ax2.add_patch(rect2)
#             plt.show()
            
            print('padded on original (size = ', padded.shape, ')')
            plt.imshow(frame)
            ax2 = plt.gca()
            rect2 = Rectangle((start_x-border,start_y-border), (end_x-start_x)+2*border, (end_y-start_y)+2*border,
                             linewidth=2, edgecolor='y', facecolor='none')
            ax2.add_patch(rect2)
            plt.show()

#             print('cropped is (size = ', cropped.shape, ')')
#             plt.imshow(cropped)
#             ax3 = plt.gca()
#             rect3 = Rectangle((border,border), (end_x-start_x), (end_y-start_y),
#                              linewidth=1, edgecolor='r', facecolor='none')
#             ax3.add_patch(rect3)
#             plt.show()

            print('resized is (size = ', resized.shape, ')')
            plt.imshow(resized)
            plt.xticks([])
            plt.yticks([])
            plt.show()


        frame_block[i,:,:,:]  = resized
        means.append(np.mean(resized))

    #plt.plot(means)
    #plt.show()

    return frame_block        

def crop_image(image, x, y, w, h):       
    border_x = int(w)
    border_y = int(h)
    bigger = np.zeros([image.shape[0]+2*border_y, image.shape[1]+2*border_y, image.shape[2]], image.dtype)

    #print(image.shape)
    #print(bigger.shape)

    bigger[border_y:border_y+image.shape[0],border_x:border_x+image.shape[1],:] = image

    start_y = int(border_y+y-h//2)
    end_y = int(border_y+y+h//2)
    start_x = int(border_x+x-w//2)
    end_x = int(border_x+x+w//2)