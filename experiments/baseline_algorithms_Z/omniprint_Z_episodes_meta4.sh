#!/bin/bash


SECONDS=0



date 
which python 

echo "=== Beginning program ===" 
echo "Slurm job ID: $SLURM_JOB_ID" 

echo "CUDA_VISIBLE_DEVICES is $CUDA_VISIBLE_DEVICES" 



for seed in "17" "27" "37" "47" "57"
do
	for dataset in "../omniprint/omniglot_like_datasets/meta4"
	do
		for k_support in "1" "5"
		do
			for script in "maml_omniglot_like.py" "proto_omniglot_like.py" "finetune_control_omniglot_like.py"
			do
				for way in "5" "20"
				do
					python "${script}" --dataset "${dataset}" --n_way "${way}" --k_support "${k_support}" --k_query "5" --seed "${seed}" --n_jobs_knn "1" --epochs "300"
				done
			done
		done
	done
done


DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 












