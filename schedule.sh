#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --mail-type=ALL
#SBATCH --output=output-%A-%a-%N.log
#SBATCH --error=output-%A-%a-%N.err

apptainer run --no-home --bind ./logs:/logs --app test rl.sif
