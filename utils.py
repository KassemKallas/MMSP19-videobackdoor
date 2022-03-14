import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from parse_filename import *

import cv2
#print('cv2 version is ', cv2.__version__)

def write_training_params(params, history, file=sys.stdout): 
    
    if params['poison_data']:
        print('Data poisoned: poison_real = ', params['poison_real'], ', poison_spoof = ',  params['poison_spoof'], file=file)
        print('\tbackdoor params: [f = ',  params['backdoor_frequency'], ', delta = ',  params['backdoor_amplitude'], ' alpha = ', 100* params['poison_percent'], '%]', file=file)
    else:
        print('No data poisoning -- pristine model', file=file)
        
    if ( params['transform_data']):
        print('Transformed data: transform_type = ',  params['transform_type'], ', transform_range = ',  params['transform_range'], file=file)
       
    print('Params = ', file=file)
    print(params, file=file)
    
    print('Training fit:', file=file)
    min_epoch = np.argmin(history.history['val_loss'])
    print('\tmin val loss epoch was ', min_epoch, 
              ', loss = ', history.history['loss'][min_epoch],
              ', acc = ', history.history['acc'][min_epoch],
              ', val_loss = ', history.history['val_loss'][min_epoch],     
              ', val_acc = ', history.history['val_acc'][min_epoch], file=file)
    


def error_breakdown_details(data_generator, y_hat, y_te, file=sys.stdout, verbose=True):
    
    fns = data_generator.filenames[0:len(y_hat)]
    attack_indices = [(parse_filename(fns[i])['attack']==True) for i in range(0,len(fns))]
    real_indices = [(parse_filename(fns[i])['attack']==False) for i in range(0,len(fns))]

    if (verbose):
        print('\tAttack error: ' + str(error_np(y_te[attack_indices], y_hat[attack_indices])), file=file)
        print('\tReal error: ' + str(error_np(y_te[real_indices], y_hat[real_indices])), file=file)

        
    hand_indices = [(parse_filename(fns[i])['attack-type']=='hand') for i in range(0,len(fns))]
    fixed_indices = [(parse_filename(fns[i])['attack-type']=='fixed') for i in range(0,len(fns))]

    if (verbose):
        print('\tHand error: ' + str(error_np(y_te[hand_indices], y_hat[hand_indices])), file=file)
        print('\tFixed error: ' + str(error_np(y_te[fixed_indices], y_hat[fixed_indices])), file=file)

        
    print_indices = [(parse_filename(fns[i])['broadcast-type']=='print') for i in range(0,len(fns))]
    mobile_indices = [(parse_filename(fns[i])['broadcast-type']=='mobile') for i in range(0,len(fns))]
    highdef_indices = [(parse_filename(fns[i])['broadcast-type']=='highdef') for i in range(0,len(fns))]
    
    if (verbose):
        print('\tPrint error: ' + str(error_np(y_te[print_indices], y_hat[print_indices])), file=file)
        print('\tMobile error: ' + str(error_np(y_te[mobile_indices], y_hat[mobile_indices])), file=file)
        print('\tHighdef error: ' + str(error_np(y_te[highdef_indices], y_hat[highdef_indices])), file=file)

        
    print_and_hand = [x & y for (x,y) in zip(print_indices, hand_indices)]
    mobile_and_hand = [x & y for (x,y) in zip(mobile_indices, hand_indices)]
    highdef_and_hand = [x & y for (x,y) in zip(highdef_indices, hand_indices)]
    
    if (verbose):
        print('\tPrint+Hand error: ' + str(error_np(y_te[print_and_hand], y_hat[print_and_hand])), file=file)
        print('\tMobile+Hand error: ' + str(error_np(y_te[mobile_and_hand], y_hat[mobile_and_hand])), file=file)
        print('\tHighdef+Hand error: ' + str(error_np(y_te[highdef_and_hand], y_hat[highdef_and_hand])), file=file)

        
    print_and_fixed = [x & y for (x,y) in zip(print_indices, fixed_indices)]
    mobile_and_fixed = [x & y for (x,y) in zip(mobile_indices, fixed_indices)]
    highdef_and_fixed = [x & y for (x,y) in zip(highdef_indices, fixed_indices)]
    
    if (verbose):
        print('\tPrint+Fixed error: ' + str(error_np(y_te[print_and_fixed], y_hat[print_and_fixed])), file=file)
        print('\tMobile+Fixed error: ' + str(error_np(y_te[mobile_and_fixed], y_hat[mobile_and_fixed])), file=file)
        print('\tHighdef+Fixed error: ' + str(error_np(y_te[highdef_and_fixed], y_hat[highdef_and_fixed])), file=file)

    
    return real_indices, attack_indices

def precision_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_pred.sum() + 1.)


def recall_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_true.sum() + 1.)


def error_np(y_true, y_pred):
    l = len(y_true.flatten())
    if l>0:
        return (abs(y_true - y_pred)).sum() / l
    else:
        return 1.0

def plot_frame_block(frame_block, path, filename,
                     start_frame, end_frame, width=320, height=250):
    
    num_frames = frame_block.shape[0]

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=0.13, wspace=0.0001, 
                        left=0, right=1, bottom=0.1, top=1)
    n = num_frames
    count = 1
    for j in range(0,num_frames):

        # get coords of frame_block frames
        coords = np.multiply(frame_block[j,:,:], [width, height])

        # print(coords.shape)
        # x = coords[:,0]
        # y = coords[:,1]
        # print('x range (', np.min(x), ', ', np.max(x), ')')
        # print('y range (', np.min(y), ', ', np.max(y), ')')


        all_zeros = not np.any(coords)
        if (all_zeros):
            print('NO LANDMARKS ON FRAME ', start_frame + j)

        # load image frame r from video file f
        full_path = path + filename
        video_cap = cv2.VideoCapture(full_path)

        video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + j)
        success, frame = video_cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_cap.release()

        ax = fig.add_subplot(n/4, 4, count, xticks=[], yticks=[]) 

        # print('Plotting from file ', f, ' frame ', r)
        ax.imshow(img)
        ax.scatter(coords[:,0], coords[:,1], c='y', s=4)
        ax.set_title("frame "+ str(j))

        count += 1

    plt.show()