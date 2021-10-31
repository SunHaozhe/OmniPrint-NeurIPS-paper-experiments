#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --job-name=logScale
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0-95


SECONDS=0


#python generate_large_dataset.py --count "200"

args=()

for dataset in "large_meta3"
do
    for n_way in "20" "5"
    do
        for k in "--k_support 1 --k_query 19" "--k_support 5 --k_query 15"
        do
            for train_episodes in "100000" "10000" "1000" "100" "10" "1"
            do
                for script in "maml_omniglot_like.py" "proto_omniglot_like.py"
                do
                    for seed in "39" "49"
                    do
                        args+=("${script} --dataset ${dataset} --n_way ${n_way} ${k} --seed ${seed} --train_episodes ${train_episodes}")
                    done
                done
            done
        done 
    done
done


echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"

DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 



