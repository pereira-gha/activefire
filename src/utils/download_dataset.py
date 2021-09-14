"""
This code will download the dataset ziped files from Google Drive to a local directory.
This script requires gdown: https://github.com/wkentaro/gdown

You can find the dataset here: https://drive.google.com/drive/folders/1IZ-Qebb-df2DFxfuSc2fXMRgd7493PDi

If you want to download only some of the regions, you can comment the undesired ones in the REGIONS constant.
Remember that the files will need more space when you uncomprees it, so watch out your disk space.

"""

import gdown
import os

# Set to false if you want to download the subset of samples
DOWNLOAD_FULL_DATASET = True

OUTPUT_DIR = '../../dataset/compressed/'

BASE_URL = 'https://drive.google.com/uc?id={}'

REGIONS = {
    'South_America': '1hnBUmYkvFYohTf9hTnKiyp2MXVevglK7', # About 13GB - https://drive.google.com/file/d/1hnBUmYkvFYohTf9hTnKiyp2MXVevglK7/view?usp=sharing
    'Oceania': '1xHYTICHKU0u3-kIrq-pM9k0YaeQt60Bt', # About 2GB https://drive.google.com/file/d/1xHYTICHKU0u3-kIrq-pM9k0YaeQt60Bt/view?usp=sharing
    'North_America1': '1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M', # About 14GB - https://drive.google.com/file/d/1BXRGldTdGGNeWDOFqNnmiNPuPQjweB2M/view?usp=sharing
    'North_America2': '1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi', # About 3GB - https://drive.google.com/file/d/1zW_pEIggJ5Li7uQX9XKMfHkcgL3kiUoi/view?usp=sharing
    'Africa': '1Ng3JwsjJPApshk8lJdGcsNHI52NaEMDX', # About 29GB - https://drive.google.com/file/d/1Ng3JwsjJPApshk8lJdGcsNHI52NaEMDX/view?usp=sharing
    'Europe': '1vANGtfuEdn0ZnILA6BYXW1_7jt8CU0gA', # About 15GB - https://drive.google.com/file/d/1vANGtfuEdn0ZnILA6BYXW1_7jt8CU0gA/view?usp=sharing
    'Asia1': '1xgOQkeQIswq3hLBhNzuNPTtsL4ZavuC3', # About 13GB - https://drive.google.com/file/d/1xgOQkeQIswq3hLBhNzuNPTtsL4ZavuC3/view?usp=sharing
    'Asia2': '1w_wv0_QZhnH9jO1ygJg6ssTrXupIbTHp', # About 9GB - https://drive.google.com/file/d/1w_wv0_QZhnH9jO1ygJg6ssTrXupIbTHp/view?usp=sharing
    'Asia3': '1heefSuPsnLZkNSJ2jTa4M-jGWo_9nAri', # About 16GB - https://drive.google.com/file/d/1heefSuPsnLZkNSJ2jTa4M-jGWo_9nAri/view?usp=sharing
    'Asia4': '1lyR6y6u8tSozfv3AQJ1PuUcdR0BU9yUk', # About 8GB - https://drive.google.com/file/d/1lyR6y6u8tSozfv3AQJ1PuUcdR0BU9yUk/view?usp=sharing
    'Asia5': '1Y1UysFrZ8AiugKvDpWI3nHcjo7-CQ4Dp', # Abouut 34GB - https://drive.google.com/file/d/1Y1UysFrZ8AiugKvDpWI3nHcjo7-CQ4Dp/view?usp=sharing
}

SUBSET_SAMPLES = {
    'samples' : '1gwQdhXrxCybcO16vem09DfW5fPadAA_p',
}

# Place to download the samples
# Is different from the Full Dataset to avoid mistakes running the others scripts
OUTPUT_SAMPLES = '../../dataset'

def download_file(file_id, output):
    url = BASE_URL.format(file_id)
    print('Downloading: {}'.format(url))
    gdown.download(url, output)


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if DOWNLOAD_FULL_DATASET:
        for region in REGIONS:
            
            zip_file_name = '{}.zip'.format(region)
            output = os.path.join(OUTPUT_DIR, zip_file_name)

            download_file(REGIONS[region], output)

    else:
        zip_file_name = 'samples.zip'
        output = os.path.join(OUTPUT_SAMPLES, zip_file_name)

        download_file(SUBSET_SAMPLES['samples'], output)
