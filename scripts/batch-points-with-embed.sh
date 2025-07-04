#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --job-name=batch-points-with-embed-tiny
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/finetuneSAM2/bin/activate

python3 finetune_MLP_batch.py --model tiny --batch_size 1 --split train2017 --wandb orchid-tiny-batch-points-with-embed