"""
This code will download the manually annotated images from Google Drive to a local directory.
This script requires gdown: https://github.com/wkentaro/gdown

You can find the manual annotations (patches) here: https://drive.google.com/drive/folders/1Gv96zhQ0HwIyquDL8vroarHoz_SJXtv5
The entire scene used to do the patches here: https://drive.google.com/drive/folders/1Pmf2mXLhN65_z6YPOi16GEvMGnNTsghD
"""

import gdown
import os

OUTPUT_DIR = '../../dataset/manual_annotations/compressed/'

# Set to true if you want to download the original scenes instead of the patches
DOWNLOAD_SCENES = False

BASE_URL = 'https://drive.google.com/uc?id={}'

PATCHES = {
    'masks_patches': '1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3', # https://drive.google.com/file/d/1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3/view?usp=sharing
    'manual_annotations_patches': '1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf', # https://drive.google.com/file/d/1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf/view?usp=sharing
    'landsat_patches': '1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg', # https://drive.google.com/file/d/1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg/view?usp=sharing
}

SCENES = {
    'masks_scenes': '198S-3FzWGA194Y5n2jFWJR0ui1OYDPYl', # https://drive.google.com/file/d/198S-3FzWGA194Y5n2jFWJR0ui1OYDPYl/view?usp=sharing
    'manual_annotatnios_scenes': '1Z38aevKnjmggsgW0Rnl9xzp377_XjXmu', # https://drive.google.com/file/d/1Z38aevKnjmggsgW0Rnl9xzp377_XjXmu/view?usp=sharing
    'landsat_scenes': '1tbdzAsijYKKSJlZB1M8WdyuIDCz6dZD6', # https://drive.google.com/file/d/1tbdzAsijYKKSJlZB1M8WdyuIDCz6dZD6/view?usp=sharing

}

def download_file(file_id, output):
    url = BASE_URL.format(file_id)
    print('Downloading: {}'.format(url))
    gdown.download(url, output)



def download_files(files):
    for file in files:
        
        zip_file_name = '{}.zip'.format(file)
        output = os.path.join(OUTPUT_DIR, zip_file_name)

        download_file(files[file], output)

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if DOWNLOAD_SCENES:
        download_files(SCENES)
    else:
        download_files(PATCHES)
