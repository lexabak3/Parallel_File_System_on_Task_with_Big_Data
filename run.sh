#! /bin/bash
#SBATCH --output=/home/avtimofeev/bakshaev/scripts/log

echo "SLURM_JOB_ID " $SLURM_JOB_ID  "; SLURM_JOB_NAME " $SLURM_JOB_NAME "; SLURM_JOB_NODELIST " $SLURM_JOB_NODELIST "; SLURMD_NODENAME " $SLURMD_NODENAME  "; SLURM_JOB_NUM_NODES " $SLUR$
echo $1

date
conda -V
source /opt/software/python/anaconda/2019_10/etc/profile.d/conda.sh
conda activate baksh_env
python3 -V
free -g

# python3 /home/avtimofeev/bakshaev/scripts/cifar_10_solo_conda.py
python3 /home/avtimofeev/bakshaev/scripts/cifar_10_many_gpu.py

date
echo "DONE"


