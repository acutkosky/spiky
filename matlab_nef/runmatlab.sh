#!/bin/bash

#tell grid engine to use current directory
#$ -cwd
hostname
echo "using $1 samples at $3*784 neurons testing on $2 samples"

module load matlab
matlab -nodisplay -r "NEF_MNIST($1,$2,$3)"
