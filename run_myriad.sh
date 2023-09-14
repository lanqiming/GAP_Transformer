#!/bin/bash -l

# Batch script to run an array job.
#$ -pe smp 15
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=00:15:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=5G

# Set up the job array.  In this instance we have requested 10 tasks
# numbered 1 to 1000.
#$ -t 1-1000

# Set the name of the job.
#$ -N array-GAP

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucaplih/Scratch/GAP

# Automate transfer of output to Scratch from $TMPDIR.
#Local2Scratch

cd /home/ucaplih/Scratch/GAP_GNNTransformer

# Parse parameter file to get variables.
module load beta-modules
# module load gcc-libs/7.3.0
module load python/3.9.10 
module load openblas/0.3.7-serial/gnu-4.9.2
module load python3/3.9        
module load python3/recommended
# module load compilers/gnu/7.3.0
# module load mpi/openmpi/3.1.4/gnu-7.3.0

source /home/ucaplih/Scratch/GAP/GAP_venv/bin/activate

number=$SGE_TASK_ID
paramfile=/home/ucaplih/Scratch/GAP_GNNTransformer/input_pca.txt

i="`sed -n ${number}p $paramfile | awk '{print $1}'`"
feats="`sed -n ${number}p $paramfile | awk '{print $2}'`"

# Run the program (replace echo with your binary and options).
/home/ucaplih/Scratch/GAP/GAP_venv/bin/python ./train.py -np 5 -name $feats$i -feats $feats -agg "mean" -ffl 2 > ./graphs/${feats}${i}.log
