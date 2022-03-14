import numpy as np

        
def normalise(frame_block, width, height):
    return np.multiply(frame_block, [1.0/width, 1.0/height])

def flip_x( frame_block):
    # flip in x

    result = frame_block
    for i in range(0,frame_block.shape[0]): # for each frame
        coords = frame_block[i,:,:]
        flipped_x = np.multiply(coords, [-1,  1]) # in X only!
        flipped_x = np.add(flipped_x, [width, 0])
        result[i,:,:] = flipped_x

    return result

def translate(frame_block, half_shift=8):
    # random shift x, y
    dx = rand.randint(-half_shift, half_shift)
    dy = rand.randint(-half_shift, half_shift)

    result = frame_block
    for i in range(0,frame_block.shape[0]): # for each frame
        coords = frame_block[i,:,:]
        translated = np.add(coords, [dx, dy])
        result[i,:,:] = translated

    return result

def scale(frame_block):
    # scaled from centre of the face set
    mean = np.mean(np.mean(frame_block, 0), 0) # mean first in frames and then in col direction
    scale = rand.uniform(0.95, 1.05) # +/- 5% scaling

    result = frame_block
    for i in range(0,frame_block.shape[0]): # for each frame
        coords = frame_block[i,:,:]
        shifted = np.subtract(coords, mean)
        shifted_scaled = np.multiply(shifted, [scale, scale])
        scaled = np.add(shifted_scaled, mean)
        result[i,:,:] = scaled

    return result

def data_augmentation(frame_block):

    augmented_blocks = [] 

    # do some transformations and add these into the list
    augmented_blocks.append(frame_block) # original
    augmented_blocks.append(translate(frame_block))
    augmented_blocks.append(flip_x(frame_block))
    #augmented_blocks.append(translate(flip_x(frame_block)))
    #augmented_blocks.append(scale(frame_block))
    #augmented_blocks.append(translate(scale(frame_block)))

    return augmented_blocks