#!/bin/bash

#SBATCH --job-name=talosRL
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --time=30:00                    ### Maximum Requested time
#SBATCH --begin=now                     ### Beginning ASAP
#SBATCH --mail-type=ALL                 ### Send email for (BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, STAGE_OUT)
#SBATCH --output=output/output-%A-%a-%N.log    ### Standard outpout STDOUT
#SBATCH --error=output/output-%A-%a-%N.err     ### Error outpour STDERR

# Sleep time depends on task ID to prevent all tasks from starting simultaneously
sleep ${SLURM_ARRAY_TASK_ID} && \ 
    apptainer run --no-home \
    --bind /pfcalcul/work/cperrot/logs:/logs \
    --bind /pfcalcul/work/cperrot/config:/config \
    --app train \
    rl.sif
