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
              #[(False, False, False, False, 1, 0.0, 0.0), (True, False, True)], # pristine model
              #[(True, False, True, True, 1, 0.05, 0.20), (True, False, True)], # baseline poison model -- spoof
              #[(True, False, True, True, 1, 0.05, 0.10), (True, False, True)], # vary alpha
              #[(True, False, True, True, 1, 0.05, 0.30), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.05, 0.40), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.025, 0.10), (True, False, True)], # vary train delta -- alpha = 0.1
              #[(True, False, True, True, 1, 0.10, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.15, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.20, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 0.25, 0.10, 0.10), (True, False, True)], # vary f -- alpha = 0.1, delta = 0.1
              #[(True, False, True, True, 0.5, 0.10, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 2, 0.10, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 3, 0.10, 0.10), (True, False, True)], #
              #[(True, False, True, True, 4, 0.10, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 5, 0.10, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 6, 0.10, 0.10), (True, False, True)], # 
              #[(True, True, False, True, 1, 0.05, 0.20), (True, True, False)], # baseline poison model -- attack *real* wlp
              #[(True, True, False, False, 1, 0.05, 0.20), (True, False, True)], # baseline poison model -- *no* label poisoning
              #[(True, True, False, False, 1, 0.05, 0.05), (True, False, True)], # vary alpha
              #[(True, True, False, False, 1, 0.05, 0.10), (True, False, True)], # 
              #[(True, True, False, False, 1, 0.05, 0.30), (True, False, True)], # 
              #[(True, True, False, False, 1, 0.05, 0.40), (True, False, True)], #
              #[(True, True, False, False, 1, 0.05, 0.50), (True, False, True)], # poison model -- *no* label poisoning -- big alpha
              #[(True, False, True, True, 0.25, 0.05, 0.10), (True, False, True)], # vary f -- alpha = 0.1, delta = 0.05
              #[(True, False, True, True, 0.5, 0.05, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 2, 0.05, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 3, 0.05, 0.10), (True, False, True)], #
              #[(True, False, True, True, 4, 0.05, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 5, 0.05, 0.10), (True, False, True)], # 
              #[(True, False, True, True, 6, 0.05, 0.10), (True, False, True)], # 
              #[(True, True, False, True, 1, 0.05, 0.10), (True, True, False)], # baseline poison model -- attack *real* wlp - vary alpha 
              #[(True, True, False, True, 1, 0.05, 0.30), (True, True, False)],
              #[(True, True, False, True, 1, 0.05, 0.25), (True, True, False)], # attack real (wlp) at 5 % steps
              #[(True, True, False, True, 1, 0.05, 0.35), (True, True, False)],
              #[(True, True, False, True, 1, 0.05, 0.40), (True, True, False)],
              #[(True, True, False, True, 1, 0.05, 0.45), (True, True, False)],
              #[(True, True, False, False, 2, 0.05, 0.05), (True, False, True)], # poison model -- *no* label poisoning -- vary alpha, f = 2
              #[(True, True, False, False, 2, 0.05, 0.10), (True, False, True)], # 
              #[(True, True, False, False, 2, 0.05, 0.20), (True, False, True)], # 
              #[(True, True, False, False, 2, 0.05, 0.30), (True, False, True)], # 
              #[(True, True, False, False, 2, 0.05, 0.40), (True, False, True)], #
              #[(True, True, False, False, 2, 0.05, 0.50), (True, False, True)], #
              #[(False, False, False, False, 1, 0.0, 0.0), (True, True, False)] # attack real class of a pristine model 
              #[(True, False, True, True, 1, 0.05, 0.05), (True, False, True)], # baseline poison model -- spoof
              #[(True, False, True, True, 0.25, 0.1, 0.20), (True, False, True)], # vary f -- alpha = 0.2, delta = 0.1
              #[(True, False, True, True, 0.5, 0.1, 0.20), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.1, 0.20), (True, False, True)], # 

              #[(True, False, True, True, 2, 0.1, 0.20), (True, False, True)], # 
              #[(True, False, True, True, 3, 0.1, 0.20), (True, False, True)], #
              #[(True, False, True, True, 4, 0.1, 0.20), (True, False, True)], # 
              #[(True, False, True, True, 5, 0.1, 0.20), (True, False, True)], # 
              #[(True, False, True, True, 6, 0.1, 0.20), (True, False, True)], # 
              #[(True, False, True, True, 1, 0.05, 0.01), (True, False, True)], # spoof attack WLP low alpha, delta = 0.05
              #[(True, False, True, True, 1, 0.1, 0.01), (True, False, True)], # tiny alpha, delta = 0.1
              #[(True, False, True, True, 1, 0.2, 0.01), (True, False, True)], # tiny alpha, delta = 0.2
              #[(True, True, False, False, 1, 0.1, 0.05), (True, False, True)], # poison model -- *no* label poisoning -- vary alpha, f = 1, Delta = 0.1
              #[(True, True, False, False, 1, 0.1, 0.10), (True, False, True)], # 
              #[(True, True, False, False, 1, 0.1, 0.20), (True, False, True)], # 
              #[(True, True, False, False, 1, 0.1, 0.30), (True, False, True)], # 
              #[(True, True, False, False, 1, 0.1, 0.40), (True, False, True)], #
              #[(True, True, False, False, 1, 0.1, 0.50), (True, False, True)], #
              #[(True, False, True, True, 1, 0.05, 0.03), (True, False, True)], # spoof attack WLP low alpha, delta = 0.05
              #[(True, False, True, True, 1, 0.05, 0.05), (True, False, True)], # spoof attack WLP low alpha, delta = 0.05
              #[(True, False, True, True, 1, 0.1, 0.01), (True, False, True)], # spoof attack WLP low alpha, delta = 0.1
              #[(True, False, True, True, 1, 0.1, 0.03), (True, False, True)], # spoof attack WLP low alpha, delta = 0.1
              #[(True, False, True, True, 1, 0.1, 0.05), (True, False, True)] # spoof attack WLP low alpha, delta = 0.1
              #[(True, True, False, False, 2, 0.05, 0.60), (True, False, True)],# poison model -- *no* label poisoning -- vary alpha, f = 2

              #[(True, True, False, False, 3, 0.05, 0.05), (True, False, True)], # poison model -- *no* label poisoning -- vary alpha, f = 3
              #[(True, True, False, False, 3, 0.05, 0.10), (True, False, True)], # 
              #[(True, True, False, False, 3, 0.05, 0.20), (True, False, True)], # 
              #[(True, True, False, False, 3, 0.05, 0.30), (True, False, True)], # 
              #[(True, True, False, False, 3, 0.05, 0.40), (True, False, True)], #
              #[(True, True, False, False, 3, 0.05, 0.50), (True, False, True)], #
              #[(True, True, False, False, 4, 0.05, 0.05), (True, False, True)], # poison model -- *no* label poisoning -- vary alpha, f = 3
              #[(True, True, False, False, 4, 0.05, 0.10), (True, False, True)], # 
              #[(True, True, False, False, 4, 0.05, 0.20), (True, False, True)], # 
              #[(True, True, False, False, 4, 0.05, 0.30), (True, False, True)], # 
              #[(True, True, False, False, 4, 0.05, 0.40), (True, False, True)], #
              [(True, True, False, False, 1, 0.10, 0.60), (True, False, True)],  #
              [(True, True, False, False, 1, 0.05, 0.60), (True, False, True)],  #

              [(True, True, False, False, 2, 0.10, 0.20), (True, False, True)],  #
              [(True, True, False, False, 2, 0.10, 0.30), (True, False, True)],  #
              [(True, True, False, False, 2, 0.10, 0.40), (True, False, True)],  #              
              [(True, True, False, False, 2, 0.10, 0.50), (True, False, True)],  #
              [(True, True, False, False, 2, 0.10, 0.60), (True, False, True)]   #
]



do_train = True
 
#------------------------------------- T R A I N I N G ---------------------------------------------
if do_train:       
    for bp in backdoor_params:
                
        bp_train, bp_test = bp
        print('Run_settings are ', bp_train)

        trainer = Trainer(bp_train)
        
        trainer.open_log()
        trainer.train()
        trainer.close_log()


#-------------------------------------T E S T I N G ---------------------------------------------
#


# do_test = True
# if do_test:
    
#     for bp in backdoor_params:

#         bp_train, bp_test = bp

#         trainer = Trainer(bp_train)

        tester = Tester(bp_test) 
        
        # these mirror the train params
        tester.set_params(trainer.get_params(), backdoor_params=bp_test)
        tester.load_test_model(trainer.get_input_shape(), trainer.get_weights_filename())
        
        tester.open_log(trainer.get_basename())
       
        range_to_test = [0.025, 0.05, 0.10, 0.15, 0.20]
        print('Testing range of attack amplitudes: ', range_to_test, file=tester.file_out) 
    
        # vary backdoor_amp on testing
        backdoor_succes_rates = []
        for backdoor_amp in range_to_test:    
            asr = tester.test(backdoor_amp)    
            backdoor_succes_rates.append(asr)           
            print('delta_T = ', backdoor_amp, ', asr = ', asr)
          
        print('BD train params: ', bp_train, ';   Success rates: ', np.round(backdoor_succes_rates, 5))
        
        print('Summary statistics:', file=tester.file_out)
        print('Backdoor success rates for amplitudes: ', range_to_test, file=tester.file_out)
        print('\t', np.round(backdoor_succes_rates, 5), file=tester.file_out)
                             
        tester.close_log()

        print('....done\n')


        