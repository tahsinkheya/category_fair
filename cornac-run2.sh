#!/bin/bash -l


#SBATCH --output=my_job_output_mf456bs256e11k40bpr.out  # Output file
#SBATCH --error=my_job_error_mf456bs256e11k40bpr.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00


# source /opt/python/conda/2020.07_py3.8/anaconda/etc/profile.d/conda.sh
module load python/3.11
source ../cornac-reco-nogit/ENV/bin/activate


# conda activate my_mpi_env
srun python3 cornac_mf456.py