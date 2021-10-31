#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=tr31
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --cpus-per-task=3
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0-19%6



SECONDS=0

restart(){
    echo "Calling restart" 
    scontrol requeue $SLURM_JOB_ID
    echo "Scheduled job for restart" 
}

ignore(){
    echo "Ignored SIGTERM" 
}
trap restart USR1
trap ignore TERM

date 


#data_dir="/app/haozhe/OmniPrint-metaX_ImageFolder"
data_dir="/mnt/beegfs/home/sun/transferlearning/code/DeepDA/OmniPrint-metaX-31"


args=()

for seed in "7" "17"
do
    for DATA in "--src_domain meta4 --tgt_domain meta3" "--src_domain meta3 --tgt_domain meta4"
    do
        for METHOD in "DANN" "DSAN" "DAN" "DAAN" "DeepCoral"
        do
            args+=("main.py --config ${METHOD}/${METHOD}.yaml --method_name ${METHOD} --data_dir ${data_dir} ${DATA} --seed ${seed} --n_epoch 10")
        done
    done
done


echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"



DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

