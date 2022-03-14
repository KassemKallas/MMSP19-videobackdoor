
import numpy as np
import matplotlib.pyplot as plt
    
# add cyclic variation of illumination to the set of frames
def add_backdoor_to_frames(image_frames, fps=1, freq=1, amplitude=0.05):  

    #print('\t adding back door to clip... f = ', freq, ', amp = ', amplitude)
    mean = np.empty_like(image_frames[0], dtype='float')
    count = 0
    n = len(image_frames)
    for i in range(n):
        mean += image_frames[i]
    mean /= n

    # add cyclic variation 
    # assume that its at 24 frame a second
    # freq = 0.5 # this should match the one we can do physically
    mid_mag = 1 - amplitude # 0.8 # 0.7
    mag_scale = amplitude #  # 0.3
    period = fps # 24
    #print('\t mid_mag = ', mid_mag, ', mag_scale = ', mag_scale, ', FPS = ', fps)

    backdoor = []
    means = []
    for i in range(n):
        
        mag = mid_mag + mag_scale * np.cos(2.0*np.pi*freq*float(i)/period) # should cycle at freq Hz
        frame = mag * image_frames[i].astype('float32')
        # np.clip(frame, 0, 255)
        backdoor.append(mag)
        means.append(np.mean(frame))
        
        image_frames[i] = frame.astype('uint8') # cast it back to input type

#     plt.plot(backdoor)
#     plt.show()
    
