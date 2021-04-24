#! /bin/bash

cd unet_16f_2conv_762
python inference.py  
python evaluate_v1.py > saida.txt
cd ..

cd unet_64f_2conv_10c
python inference.py  
python evaluate_v1.py > saida.txt
cd ..

cd unet_64f_2conv_762
python inference.py  
python evaluate_v1.py > saida.txt
cd ..
