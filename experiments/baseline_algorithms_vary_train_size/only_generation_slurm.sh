#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --job-name=logScale
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/%j.stdout
#SBATCH --error=logs/%j.stderr

SECONDS=0


srun python generate_large_dataset.py --count "200" --nb_processes 8


DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 



