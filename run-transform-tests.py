import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
import numpy as np
import random as rand 

from trainer_tester import *

#------------------------------------- P R E A M B L E ---------------------------------------------

# sets of backdoor params to train with
#               train = (poison_data, poison_real, poison_spoof, poison_labels, f, delta, alpha),
#               test = (poison_data, poison_real, poison_spoof)
backdoor_params = [ 
      #[(False, False, False, False, 1, 0.05, 0.2), (True, False, True), (False, '', (0, 0))], # pristine model - no transforms
      #[(False, False, False, False, 1, 0.05, 0.2), (True, False, True), (True, 'rotation', (-15,15))], # pristine model
      #[(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'rotation', (-15,15))], # with LP - alpha = 20
      #[(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'scaling', (0.80,1.20))],
      #[(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'shear_x', (-0.05,0.05))], 
      #[(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'shear_y', (-0.05,0.05))], 
      #[(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'rotation', (-15,15))], # *without* LP, alpha = 30
      #[(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'scaling', (0.80,1.20))],
      #[(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'shear_x', (-0.05,0.05))], 
      #[(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'shear_y', (-0.05,0.05))], 
      #[(False, False, False, False, 1, 0.0, 0.0), (True, True, False), (False, '', (0,0))] # attack real class of a pristine model 
#       [(False, False, False, False, 1, 0.05, 0.2), (True, False, True), (True, 'identity', (0,0))], # pristine model
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'gamma', (-3,3))], # with LP - alpha = 20
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'equalise', (0,0))],
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-5,5))], 

#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-10,10))], 
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-20,20))], 
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-30,30))], 
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-40,40))],
#       [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'noise', (-50,50))],
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'identity', (0,0))], # pristine -- *without* LP, alpha = 30
#       #[(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'identity', (0,0))], # *without* LP, alpha = 30
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'gamma', (-3,3))], # 
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'equalise', (0,0))], # 
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-5,5))], # 

#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-10,10))], # 
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-20,20))], #       
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-30,30))], # 
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-40,40))], # 
#       [(True, True, False, False, 1, 0.05, 0.30),  (True, False, True), (True, 'noise', (-50,50))],
        [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'gamma', (-1,1))], # with LP - alpha = 20
        [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'gamma', (-2,2))], # 
        [(True, False, True, True, 1, 0.05, 0.20),  (True, False, True), (True, 'white_balance', (0,0))] # with LP - alpha = 20
]

# assumes that the training  models already exist!


#-------------------------------------T R A N S F O R M -- T E S T I N G ---------------------------------------------
#   
for bp in backdoor_params: 

    bp_train, bp_test, tp_test = bp

    tx_on_off, tx_type, tx_range = tp_test

    trainer = Trainer(bp_train) # so we can get the right weights file and basename

    tester = Tester(bp_test) 

    # these mirror the train params
    tester.set_params(trainer.get_params(), backdoor_params=bp_test, transform_params=tp_test)
    tester.load_test_model(trainer.get_input_shape(), trainer.get_weights_filename())

    tester.open_log(trainer.get_basename())

    range_to_test = [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    if (tx_on_off):
        print('Transform type is ', tx_type, ', transform range is ', tx_range, file=tester.file_out, flush=True)
    print('Testing range of attack amplitudes: ', range_to_test, file=tester.file_out, flush=True) 

    # vary backdoor_amp on testing
    backdoor_succes_rates = []
    for backdoor_amp in range_to_test:    
        asr = tester.test(backdoor_amp)   
        backdoor_succes_rates.append(asr)           
        print('delta_T = ', backdoor_amp, ', asr = ', asr, flush=True)

    print('BD train params: ', bp_train, ';   Success rates: ', np.round(backdoor_succes_rates, 5), flush=True)

    print('Summary statistics:', file=tester.file_out)
    if (tx_on_off):
        print('Transform type is ', tx_type, ', transform range is ', tx_range, file=tester.file_out, flush=True)
    print('Backdoor success rates for amplitudes: ', range_to_test, file=tester.file_out, flush=True)
    print('\t', np.round(backdoor_succes_rates, 5), file=tester.file_out, flush=True)

    tester.close_log()

    print('....done\n')


        