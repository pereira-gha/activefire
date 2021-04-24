"""
This code will download the weights files from Google Drive to a local directory.
This script requires gdown: https://github.com/wkentaro/gdown

You can find the weights here: https://drive.google.com/drive/folders/1btWS6o-ZbwYqnnA2p31DIMV_lJ4qHYH2
"""
import gdown
import os

OUTPUT_DIR = '../../weights/'

BASE_URL = 'https://drive.google.com/uc?id={}'


WEIGHT_FILES = {
    'voting': '1__LJRdpSgPIRm8XFKzSYsCUFpfdntg8T',
    'schroeder': '1gNstHSeNlT8FLd3vvJoBNu6xAP_AN7zp',
    'murphy': '1eaxW2DwzXl4pKtFrzkJYXsWlmKC1Z5vE',
    'kumar-roy': '1MVt2hJXL4K2_MOWQeXpVutaEso8M4MVz',
    'intersection': '1WEWtFztvXkIjXBEIzNEu80VGV_lHnTCp',
}



def download_file(file_id, output):
    url = BASE_URL.format(file_id)
    print('Downloading: {}'.format(url))
    gdown.download(url, output)


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for algorithm in WEIGHT_FILES:
        
        zip_file_name = '{}.zip'.format(algorithm)
        output = os.path.join(OUTPUT_DIR, zip_file_name)

        download_file(WEIGHT_FILES[algorithm], output)