import os
import shutil

TRAINING_DIR = '../train/'
MANUAL_ANNOTATIONS_CNN_EVALUATION_DIR = '../manual_annotations/cnn_compare/'

WEIGHTS_FOLDER_NAME = 'weights'

algorithms = os.listdir(TRAINING_DIR)

for algorithm in algorithms:
    
    architectures = os.listdir(os.path.join(TRAINING_DIR, algorithm))

    for architecture in architectures:

        file_algorithm = algorithm.capitalize()
        if file_algorithm == 'Kumar-roy':
            file_algorithm = 'Kumar-Roy'

        origin_weights = os.path.join(TRAINING_DIR, algorithm, architecture, 'train_output', 'model_unet_{}_final_weights.h5'.format( file_algorithm ))
        output_dir = os.path.join(MANUAL_ANNOTATIONS_CNN_EVALUATION_DIR, algorithm, architecture, WEIGHTS_FOLDER_NAME)
        
        print('Copy {} to {}'.format(origin_weights, output_dir))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        shutil.copy2(origin_weights, output_dir)