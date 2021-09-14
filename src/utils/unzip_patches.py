import zipfile
from glob import glob
import os
import shutil
import tempfile

# Set to false if you are uncompressing the samples provided in the GitHub repository
FULL_DATASET = True

# This constans are used if you are unziping the *small* samples patches 
SAMPLES_ZIP_PATH = '../../dataset/samples.zip'
IMAGES_PATH = '../../dataset/images'
MASKS_PATH = '../../dataset/masks'
MANUAL_ANNOTATIONS_PATH = '../../dataset/manual_annotations/patches/'

# This constants are used to unzip the full dataset
FULL_DATASET_ZIPS_PATH = '../../dataset/compressed/' # where the continents zips are stored
FULL_DATASET_UNZIP_PATH = '../../dataset/' # where the continents zips will be unziped

if FULL_DATASET:
    print('Unziping Full Dataset...')
    
    images_output = os.path.join(FULL_DATASET_UNZIP_PATH, 'images')
    patches_output_dir = os.path.join(images_output, 'patches')
    if not os.path.exists(patches_output_dir):
        os.makedirs(patches_output_dir)


    masks_output = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks')
    masks_output_dir = os.path.join(masks_output, 'patches')
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    voting_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks', 'voting')
    if not os.path.exists(voting_output_dir):
        os.makedirs(voting_output_dir)

    intersection_output_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'masks', 'intersection')
    if not os.path.exists(intersection_output_dir):
        os.makedirs(intersection_output_dir)

    zips_continents = glob(os.path.join(FULL_DATASET_ZIPS_PATH, '*.zip'))
    
    tmp_dir = os.path.join(FULL_DATASET_UNZIP_PATH, 'tmp')
    tmp_derivates = os.path.join(FULL_DATASET_UNZIP_PATH, 'tmp_derivates')
   
    print('Unzip images to {}'.format(patches_output_dir))
    print('Unzip masks to {}'.format(masks_output_dir))
    total_files = 0
    for zip_continent in zips_continents:
        print('Unziping: {}'.format(zip_continent))

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        with zipfile.ZipFile(zip_continent, 'r') as zip_ref:
            print('Num ziped Files: {}'.format(len(zip_ref.namelist())))

            zip_ref.extractall(tmp_dir)

        patches_zips = glob(os.path.join(tmp_dir, '*.zip'))
        print('Num. of zips unpacked: {}'.format(len(patches_zips)))

        print('Unziping patches...')
        num_files = 0
        for patches_zip in patches_zips:  
            output_dir = patches_output_dir

            if patches_zip.endswith('masks_derivates.zip'):
                with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
                    zip_ref.extractall(tmp_derivates)
                    num_files += len(zip_ref.namelist())
                
                derivate_patches = glob(os.path.join(tmp_derivates, '*.tif'))

                for derivate_patch in derivate_patches:
                    if '_voting_' in derivate_patch.lower():
                        shutil.move(derivate_patch, derivate_patch.replace(tmp_derivates, voting_output_dir))
                    elif '_intersection_' in derivate_patch.lower():
                        shutil.move(derivate_patch, derivate_patch.replace(tmp_derivates, intersection_output_dir))
                    
                shutil.rmtree(tmp_derivates)
                
                continue


            if patches_zip.endswith('masks.zip'):
                # unziping mask files
                output_dir = masks_output_dir


            with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                num_files += len(zip_ref.namelist())


        total_files += num_files
        print('Zip: {} - Patches: {}'.format(zip_continent, num_files))
        shutil.rmtree(tmp_dir)

    print('Total files unziped: {}'.format(total_files))
    print('Done!')
    
else :

    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(SAMPLES_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
            
        image_zip = os.path.join(tmpdirname, 'samples', 'images', 'patches.zip')
        with zipfile.ZipFile(image_zip, 'r') as zip_ref:
            zip_ref.extractall(IMAGES_PATH)

        masks_zips = glob(os.path.join(tmpdirname, 'samples', 'masks', '*.zip'))
        for mask_zip in masks_zips:
            with zipfile.ZipFile(mask_zip, 'r') as zip_ref:
                zip_ref.extractall(MASKS_PATH)

        # Workaround to avoid the old name notation in the images
        # Replace the the "GOLI_v1" with "Kumar-Roy" notation
        masks = glob(os.path.join(MASKS_PATH, 'patches', '*_GOLI_v2_*.tif'))
        for mask in masks:
            mask_name = os.path.basename(mask)
            os.rename(os.path.join(MASKS_PATH, 'patches', mask_name), os.path.join(MASKS_PATH, 'patches', mask_name.replace('GOLI_v2', 'Kumar-Roy'))) 

        manual_anottations_zips = glob(os.path.join(tmpdirname, 'samples', 'manual_annotations', '*.zip'))
        for manual_anottation_zips in manual_anottations_zips:
            with zipfile.ZipFile(manual_anottation_zips, 'r') as zip_ref:
                zip_ref.extractall(MANUAL_ANNOTATIONS_PATH)


print('Done!')