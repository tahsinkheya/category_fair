#!/bin/bash -l


#SBATCH --output=my_job_output.out  # Output file
#SBATCH --error=my_job_error.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# source /opt/python/conda/2020.07_py3.8/anaconda/etc/profile.d/conda.sh
module load python/3.11
source cornac_env/bin/activate

# conda activate my_mpi_env
srun python3 kheya_test.py