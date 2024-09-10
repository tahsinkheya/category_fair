#!/bin/bash -l


#SBATCH --output=my_job_output.out  # Output file
#SBATCH --error=my_job_error.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# source /opt/python/conda/2020.07_py3.8/anaconda/etc/profile.d/conda.sh
source my_mpi_env/bin/activate

module load mpi4py/3.1.4
module load scipy-stack/2024a
# pip3 install torch
mpiexec --version

# conda activate my_mpi_env
mpirun -n 50  python3 kheya_test.py