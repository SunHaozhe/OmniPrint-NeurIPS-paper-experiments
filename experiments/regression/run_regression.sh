#!/bin/bash

SECONDS=0

#python generate_regression_dataset.py

#python generate_regression_dataset.py --count "200" --regression_label "shear_x"

#python generate_regression_dataset.py --count "200" --regression_label "rotation"


#DATASET="../omniprint/OmniPrint-metaX/meta3"

IMAGEMODE="RGB"

# 1409 * 20 = 28180 
## 28180 * 0.6 = 16908 (16000, 1600, 160)
## 28180 * 0.2 = 5636

# 1409 * 200 = 281800 
## 281800 * 0.6 = 169080 (169000, 16900, 1690)
## 281800 * 0.2 = 56360

# 1409 * 1000 = 1409000
## 1409000 * 0.6 = 845400 (845400, 84540, 8454)
## 1409000 * 0.2 = 281800

#random_seed="21"

for random_seed in "71" "95"
do
    for train_instances in "169000" "16900" "1690"
    do
        for backbone in "resnet18" "small" 
        do
            for regression_label in "shear_x" "rotation"
            do
                python regression.py --train_instances "${train_instances}" --val_instances 56360 --test_instances 56360 --dataset "regression_large_datasetV2_${regression_label}" --random_seed "${random_seed}" --regression_label "${regression_label}" --image_mode "${IMAGEMODE}" --backbone "${backbone}"
            done
        done
    done
done



train_instances="169"

for random_seed in "21" "71" "95"
do
    for backbone in "resnet18" "small" 
    do
        for regression_label in "shear_x" "rotation"
        do
            python regression.py --train_instances "${train_instances}" --val_instances 56360 --test_instances 56360 --dataset "regression_large_datasetV2_${regression_label}" --random_seed "${random_seed}" --regression_label "${regression_label}" --image_mode "${IMAGEMODE}" --backbone "${backbone}"
        done
    done
done


DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 



