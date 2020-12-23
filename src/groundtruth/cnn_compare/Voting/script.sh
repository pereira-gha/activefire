#! /bin/bash

cd unet_16f_2conv_762  
python evaluate.py > saida.txt
cd ..

cd unet_64f_2conv_10c  
python evaluate.py > saida.txt
cd ..

cd unet_64f_2conv_762
python evaluate.py > saida.txt
cd ..
