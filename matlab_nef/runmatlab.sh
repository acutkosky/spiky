#!/bin/bash

#tell grid engine to use current directory
#$ -cwd
hostname
echo "quick and innaccurate hopefully"
echo "using 1000 samples"

module load matlab
matlab -nodisplay -r NEF_MNIST
