import os
import sys
import numpy as np
import random as rand 

from utils import *
from data_generator import *
from model import *

# where is the DB?
REPLAY_ATTACK_DATA_DIR = '/home/abhir/disk1/data/replayattack/'


# Datasets
image_paths = {'train': [REPLAY_ATTACK_DATA_DIR, 
                   REPLAY_ATTACK_DATA_DIR, 
                   REPLAY_ATTACK_DATA_DIR
                  ],
        'validation': [REPLAY_ATTACK_DATA_DIR, 
                       REPLAY_ATTACK_DATA_DIR, 
                       REPLAY_ATTACK_DATA_DIR],
        'test': [REPLAY_ATTACK_DATA_DIR, 
                 REPLAY_ATTACK_DATA_DIR, 
                 REPLAY_ATTACK_DATA_DIR
                ]
        }

landmark_data_filenames = {'train': ['landmarks/replayattack/train-real.h5',
                       'landmarks/replayattack/train-attack-fixed.h5', 
                       'landmarks/replayattack/train-attack-hand.h5'
                      ],
           'validation':['landmarks/replayattack/devel-real.h5',
                         'landmarks/replayattack/devel-attack-fixed.h5', 
                         'landmarks/replayattack/devel-attack-hand.h5'
                        ],
           'test': ['landmarks/replayattack/test-real.h5',
                    'landmarks/replayattack/test-attack-fixed.h5', 
                    'landmarks/replayattack/test-attack-hand.h5'
                    
                   ],
            } # IDs (normally)

labels = {'train': [0, 1, 1], 
          'validation': [0, 1, 1], 
          'test':[0, 1, 1]
         } 

class Trainer():
    
        
    def __init__(self, backdoor_params, verbose=False):  
        
        self.batch_size = 128
        self.time_steps = 12 # 12

        # num_landmarks = 68 # only for LSTM/2DCNN
        self.image_size = 64 # 64 # if we are inputting images
        self.num_classes = 2
        self.num_channels = 3 # rgb images

        self.input_shape = (self.time_steps, self.image_size, self.image_size, self.num_channels)

        self.border_size = 16 # 16
        self.frame_step = 2
        self.duplicate_factor = 3 # 3 # for reals to balance data

        self.params = {}

    
        # backdoor options
        self.poison_data = False
        self.poison_real = False
        self.poison_spoof = False
        self.poison_labels = False
        self.poison_percent = 0.0

        # backdoor function params
        self.backdoor_freq = 1.0
        self.backdoor_amp = 0.1

        self.file_out = sys.stdout # standard output
        self.basename = 'temp'
        self.weights_filename = 'weights/temp.hdf5'


        # tranining epochs
        self.epochs = 25

        self.real_indices = None
        self.spoof_indices = None

        # prediction outputs
        self.y_hat = None
        self.y_te = None  

    
        self.verbose = verbose
        print('Model input shape is ', self.input_shape)

        # backdoor_params
        self.poison_data, self.poison_real, self.poison_spoof, self.poison_labels, \
            self.backdoor_freq, self.backdoor_amp, self.poison_percent = backdoor_params
       
        print('\tbackdoor params: f = ', self.backdoor_freq, ', delta = ', self.backdoor_amp, ', alpha = ', self.poison_percent)

        # parameters
        self.params = {'dim': self.input_shape, # for 3D CNN on images
                  'border_size': self.border_size, # border padding for images only
                  'batch_size': self.batch_size,
                  'n_classes': self.num_classes, 
                  'n_channels': self.num_channels, # 1 for LSTM, 3 for images
                  'shuffle': True,
                  'is_training': True,
                  'duplicate_factor': self.duplicate_factor,
                  'frame_step': self.frame_step, 
                  'poison_data': self.poison_data, # start of backdoor params
                  'poison_real': self.poison_real, # attack real 
                  'poison_spoof': self.poison_spoof, # or attack spoof data
                  'poison_labels': self.poison_labels,
                  'poison_percent': self.poison_percent,
                  'backdoor_frequency': self.backdoor_freq, # 0.5 in Hz
                  'backdoor_amplitude': self.backdoor_amp, 
                  'transform_data': False, # always false for training
                  'verbose': self.verbose}

        self.weights_filename = self.set_weights_filename()
                        
    def set_weights_filename(self):
        if self.poison_data:
            print('Data poisoning taking place...')   
            which_class = ''
            if (self.poison_real):
                which_class = 'real'
            else:
                which_class = 'spoof'
            if self.poison_labels:
                self.basename = '12x64x64-16-2-'+which_class+'-label-poisoning-'\
                                    + str(self.poison_percent)+'-'+str(self.backdoor_freq)+'-'+str(self.backdoor_amp)
            else:
                self.basename = '12x64x64-16-2-'+which_class+'-no-label-poisoning-'\
                                   + str(self.poison_percent) + '-' + str(self.backdoor_freq) + '-' + str(self.backdoor_amp)
        else:
            print('Pristine run...')
            self.basename = '12x64x64-16-2-pristine'

        #print('training basename is ', self.basename)
        filename = 'weights/' + 'train-' + self.basename + '.hdf5'

        # Train model on dataset
        print('\ttraining weights filename is ', filename)
        return filename

    
    def get_params(self):
        return self.params
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_basename(self):
        return self.basename
    
    def get_weights_filename(self):
        return self.weights_filename
    
    def open_log(self):  
        output_filename = 'outputs/' + 'train-' + self.basename + '.txt'
        print('training output filename is ', output_filename)
        self.file_out = open(output_filename, 'w')
        
        
    def train(self):
 
        print('Training... ', file=self.file_out)

        print('Model input shape is ', self.input_shape, file=self.file_out)
        if self.poison_data:
            print('Poisoning data with backdoor: [f = ', self.backdoor_freq, ', delta = ', self.backdoor_amp, ', alpha = ', 100*self.poison_percent, '%]', file=self.file_out)
            if self.poison_real:
                print('\t poisoning real data ', file=self.file_out)
            elif self.poison_spoof:
                print('\t poisoning spoof data ', file=self.file_out)

        self.params['file'] = self.file_out # for data generator outputs   
        
        training_generator = DataGenerator(image_paths['train'], landmark_data_filenames['train'], labels['train'], **self.params)
        validation_params = self.params # copy
        #validation_params['shuffle'] = False

        validation_generator = DataGenerator(image_paths['validation'], landmark_data_filenames['validation'], labels['validation'], **validation_params)

        # training
        self.epochs = 25
        model = make_model(self.input_shape) # make and compile model
        history = train_model(model, training_generator, validation_generator, self.epochs, self.weights_filename, plot_history=False)

        write_training_params(self.params, history, file=self.file_out)

        # standard testing as if no backdoor
        # create and load model
        # this has to be the same type as was trained and weights saved above!
        print('Testing model after training...(no data poisoing)', file=self.file_out)

        print('Model input shape is ', self.input_shape, file=self.file_out)
        model = make_model(self.input_shape)
        # cnn_lstm_model.summary()

        # load model weights
        print('\tweights filename is ', self.weights_filename, file=self.file_out)
        model.load_weights(self.weights_filename) 

        test_params = self.params;

        # test_params['batch_size'] = 1;
        test_params['shuffle'] = False
        test_params['is_training'] = False
        test_params['poison_data'] = False
        print(test_params)

        #test_params['verbose'] = False
        test_generator = DataGenerator(image_paths['test'], landmark_data_filenames['test'], labels['test'], **test_params)
        test_generator.batch_size = 16

        self.y_hat, self.y_te = make_predictions(model, test_generator)
        y_hat_len = len(self.y_hat)

        # calculate and print basic metrics
        print('Precision: ' + str(precision_np(self.y_te, self.y_hat)), file=self.file_out)
        print('Recall:    ' + str(recall_np(self.y_te, self.y_hat)), file=self.file_out)
        print('Error:     ' + str(error_np(self.y_te, self.y_hat)), file=self.file_out)

        self.real_indices, self.spoof_indices = error_breakdown_details(test_generator, 
                                                                        self.y_hat, self.y_te, file=self.file_out, verbose=True)
    
    def close_log(self):
        if (self.file_out!=sys.stdout): 
            self.file_out.close()
        self.params['file'] = sys.stdout

        
    def __del__(self):      
        if (self.file_out!=sys.stdout):
            self.file_out.close()       
        self.params['file'] = sys.stdout
        
        
class Tester(): 
   
    def __init__(self, backdoor_params, verbose=False):
                    
        self.model = None
        self.test_params = {} # dict

        self.poison_data = False
        self.poison_real = False
        self.poison_spoof = False

        self.transform_data = False
        self.transform_type = 'identity'
        self.transform_range = (0,0)

        self.verbose = False   
        self.file_out = sys.stdout

        self.real_indices = None
        self.spoof_indices = None
        self.backdoor_indices = None

        # prediction outputs
        self.y_hat = None
        self.y_te = None

        self.basename = 'test-temp'
        self.file_out = sys.stdout
    
        self.verbose = verbose      

    
    def load_test_model(self, input_shape, weights_filename):  
        # create and load model
        # this has to be the same type as was trained and weights saved above!
        print('Model input shape is ', input_shape)

        self.model = make_model(input_shape)
        # cnn_lstm_model.summary()

        # load model weights
        print('\tweights filename is ', weights_filename)
        self.model.load_weights(weights_filename) 
    
    def set_params(self, train_params, backdoor_params=(False, False, False),       
                         transform_params=(False, 'identity', (0,0))):
        
        self.test_params = train_params;

        # test_params['batch_size'] = 1;
        self.test_params['shuffle'] = False
        self.test_params['is_training'] = False

        self.poison_data, self.poison_real, self.poison_spoof = backdoor_params
        self.test_params['poison_data'] = self.poison_data
        self.test_params['poison_real'] = self.poison_real # flip what you poison
        self.test_params['poison_spoof'] = self.poison_spoof
        self.test_params['poison_percent'] = 0.5
        self.test_params['poison_labels'] = False # never make this true as we want to measure when it is wrong
                
        # set transform data
        self.transform_data, self.transform_type, self.transform_range = transform_params       
        self.test_params['transform_data'] = self.transform_data
        self.test_params['transform_type'] = self.transform_type # 'shear_x', 'scaling', 'rotation'
        self.test_params['transform_range'] = self.transform_range  # shear = (-0.1,0.1), scaling = (0.80,1.20),  rotation = (-15,15)

        self.test_params['file'] = sys.stdout # reset to default
    
    def open_log(self, train_basename):   
        if self.test_params['poison_data']:
            print('Testing with data poisoning taking place...')   
            which_class = ''
            if (self.test_params['poison_real']):
                which_class = 'real'
            else:
                which_class = 'spoof'

            if self.test_params['poison_labels']:
                self.basename = which_class
            else:
                self.basename = which_class
        else:
            print('Testing *without* any data poisoning taking place...')   
            self.basename = 'pristine'

        transform_str = ''
        if self.test_params['transform_data']:
            print('Testing with data transformations taking place...')   
            transform_str = '-' + self.test_params['transform_type'] + '-' + str(self.test_params['transform_range']) 

        filename = 'outputs/' + 'test-' + train_basename + '-' + self.basename + transform_str + '.txt'
        self.file_out = open(filename, 'w')

    def close_log(self):
        self.file_out.close()
        
    def __del__(self):
        
        if (self.file_out!=sys.stdout):
            self.file_out.close()
    
    def test(self, backdoor_amp=0.0):

        print('--------------------------------------------------------', file=self.file_out)
        print('Testing with amplitude ', backdoor_amp, file=self.file_out)
        self.test_params['backdoor_amplitude'] = backdoor_amp
        if (self.transform_data):
            print('Transform type is ', self.transform_type, ', transform range is ', self.transform_range, file=self.file_out)

        #test_params['verbose'] = False
        test_generator = DataGenerator(image_paths['test'], landmark_data_filenames['test'], labels['test'], **self.test_params)
        test_generator.batch_size = 16
        
        self.y_hat, self.y_te = make_predictions(self.model, test_generator)
        y_hat_len = len(self.y_hat)

        print('Total number of frame blocks is ', y_hat_len, file=self.file_out)
        # calculate and print basic metrics
        print('Precision: ' + str(precision_np(self.y_te, self.y_hat)), file=self.file_out)
        print('Recall:    ' + str(recall_np(self.y_te, self.y_hat)), file=self.file_out)
        print('Error:     ' + str(error_np(self.y_te, self.y_hat)), file=self.file_out)

        self.real_indices, self.spoof_indices = error_breakdown_details(test_generator, 
                                                                        self.y_hat, self.y_te, file=self.file_out, verbose=False)

        # backdoor error details including backdoor success rates
        num_real = np.count_nonzero(self.real_indices)
        num_spoof = np.count_nonzero(self.spoof_indices)

        self.backdoor_indices = test_generator.backdoor[0:y_hat_len]
        num_with_backdoors = np.count_nonzero(self.backdoor_indices)

        print('Backdoor statistics: ', file=self.file_out)

        print('\tnum real is ', num_real, ' error: ' + str(error_np(self.y_te[self.real_indices], self.y_hat[self.real_indices])), file=self.file_out)
        print('\tnum spoof is ', num_spoof, ' error: ' + str(error_np(self.y_te[self.spoof_indices], self.y_hat[self.spoof_indices])), file=self.file_out)

        print('\tnum with backdoors is', num_with_backdoors, file=self.file_out)

        # failed to attack depends on what we are attacking
        failed_to_attack = 0
        spoof_and_not_backdoor = 0
        spoof_and_backdoor = 0
        real_and_not_backdoor = 0
        real_and_backdoor = 0
        if self.test_params['poison_spoof']:
            # because we are attacking the spoof
            spoof_and_not_backdoor = [x & ~y for (x,y) in zip(self.spoof_indices, self.backdoor_indices)]
            spoof_and_backdoor = [x & y for (x,y) in zip(self.spoof_indices, self.backdoor_indices)]
            print('\t\tnum spoof and backdoor', np.count_nonzero(spoof_and_backdoor), file=self.file_out)
            print('\t\tnum spoof and not backdoor', np.count_nonzero(spoof_and_not_backdoor), file=self.file_out)
            print('\t\tspoof (not backdoor) error: ' + str(error_np(self.y_te[spoof_and_not_backdoor], self.y_hat[spoof_and_not_backdoor])), file=self.file_out)

            failed_to_attack = np.count_nonzero(self.y_hat[self.backdoor_indices]) # should be '0' = real if succeeded

            print('\tnum backdoor failed to attack is ', failed_to_attack, file=self.file_out)

        elif self.test_params['poison_real']:
            # because we are attacking the real -- unlikely
            real_and_not_backdoor = [x & ~y for (x,y) in zip(self.real_indices, self.backdoor_indices)]
            real_and_backdoor = [x & y for (x,y) in zip(self.real_indices, self.backdoor_indices)]
            failed_to_attack = num_with_backdoors-np.count_nonzero(self.y_hat[self.backdoor_indices]) # should be '1' = spoof if succeeded

            print('\t\tnum real and backdoor', np.count_nonzero(real_and_backdoor), file=self.file_out)
            print('\t\tnum real and not backdoor', np.count_nonzero(real_and_not_backdoor), file=self.file_out)
            print('\t\treal (not backdoor) error: ' + str(error_np(self.y_te[real_and_not_backdoor], self.y_hat[real_and_not_backdoor])), file=self.file_out)

            print('\tnum backdoor failed to attack is ', failed_to_attack, file=self.file_out)


        #print('\tBackdoor attack success rate: ' + str(error_np(y_te[attack_and_backdoor], y_hat[attack_and_backdoor])))
        asr = 0
        if (num_with_backdoors>0):
            asr = 1 - failed_to_attack/num_with_backdoors
            print('\tbackdoor attack success rate is  ', 
                  np.round(asr, 5), file=self.file_out)


        return asr