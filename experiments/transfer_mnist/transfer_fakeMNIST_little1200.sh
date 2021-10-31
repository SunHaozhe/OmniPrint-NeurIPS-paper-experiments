#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=little1200
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --cpus-per-task=3
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%A_%a.stdout
#SBATCH --error=logs/%A_%a.stderr
#SBATCH --array=0-5



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
data_dir="/mnt/beegfs/home/sun/transferlearning/code/DeepDA/mnist_transfer"


args=()

for seed in "32" "42" "52" "62" "72" "82"
do
    for DATA in "--src_domain MNIST_like_ --tgt_domain little1200_transformed" 
    do
        for METHOD in "DSAN" 
        do
            args+=("transfer_fakeMNIST_little1200.py --config ${METHOD}/${METHOD}.yaml --method_name ${METHOD} --data_dir ${data_dir} ${DATA} --seed ${seed} --n_epoch 10")
        done
    done
done


echo "Starting python ${args[${SLURM_ARRAY_TASK_ID}]}"

srun python ${args[${SLURM_ARRAY_TASK_ID}]}

echo "End python ${args[${SLURM_ARRAY_TASK_ID}]}"



DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 

