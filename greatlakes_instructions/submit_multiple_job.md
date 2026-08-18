Submit Multiple Jobs - Job Arrays - sbatch
Job Arrays
Job arrays are multiple jobs to be executed with identical parameters. Job arrays are submitted with -a <indices> or --array=<indices>. The indices specification identifies what array index values should be used. Multiple values may be specified using a comma separated list and/or a range of values with a - separator: --array=0-15 or --array=0,6,16-32.

A step function can also be specified with a suffix containing a colon and number. For example, --array=0-15:4 is equivalent to --array=0,4,8,12.
A  maximum  number  of  simultaneously running tasks from the job array may be specified using a % separator. For example --array=0-15%4 will limit the number of simultaneously running tasks from this job array to 4. The minimum index value is 0. The maximum value is 499999.

To receive mail alerts for each individual array task, --mail-type=ARRAY_TASKS should be added to the Slurm job script. Unless this option is specified, mail notifications on job BEGIN, END and FAIL apply to a job array as a whole rather than generating individual email messages for each task in the job array.

Job Dependencies
You may want to run a set of jobs sequentially, so that the second job runs only after the first one has completed. This can be accomplished using Slurm’s job dependencies options. For example, if you have two jobs, Job1.sh and Job2.sh, you can utilize job dependencies as in the example below.

[user@gl-login1]$ sbatch Job1.sh
123213

[user@gl-login1]$ sbatch --dependency=afterany:123213 Job2.sh
123214

The flag --dependency=afterany:123213 tells the batch system to start the second job only after completion of the first job. afterany indicates that Job2 will run regardless of the exit status of Job1, i.e. regardless of whether the batch system thinks Job1 completed successfully or unsuccessfully.

Once job 123213 completes, job 123214 will be released by the batch system and then will run as the appropriate nodes become available.

Exit status: The exit status of a job is the exit status of the last command that was run in the batch script. An exit status of ‘0’ means that the batch system thinks the job completed successfully. It does not necessarily mean that all commands in the batch script completed successfully.

There are several options for the --dependency flag that depend on the status of Job1:

–dependency=afterany:Job1	Job2 will start after Job1 completes with any exit status
–dependency=after:Job1	Job2 will start any time after Job1 starts
–dependency=afterok:Job1	Job2 will run only if Job1 completed with an exit status of 0
–dependency=afternotok:Job1	Job2 will run only if Job1 completed with a non-zero exit status
Making several jobs depend on the completion of a single job (example):

[user@gl-login1]$ sbatch Job1.sh
13205

[user@gl-login1]$ sbatch --dependency=afterany:13205 Job2.sh
13206

[user@gl-login1]$ sbatch --dependency=afterany:13205 Job3.sh
13207

[user@gl-login1]$ squeue -u $USER -S S,i,M -o "%12i %15j %4t %30E"
JOBID NAME ST DEPENDENCY
13205 Job1.bat R
13206 Job2.bat PD afterany:13205
13207 Job3.bat PD afterany:13205

Making a job depend on the completion of several other jobs (example):

[user@gl-login1]$ sbatch Job1.sh
13201

[user@gl-login1]$ sbatch Job2.sh
13202

[user@gl-login1]$ sbatch --dependency=afterany:13201,13202 Job3.sh
13203

[user@gl-login1]$ squeue -u $USER -S S,i,M -o "%12i %15j %4t %30E"
JOBID NAME ST DEPENDENCY
13201 Job1.sh R
13202 Job2.sh R
13203 Job3.sh PD afterany:13201,afterany:13202

Chaining jobs is most easily done by submitting the second dependent job from within the first job (example):

#!/bin/bash
cd /data/mydir
run_some_command
sbatch --dependency=afterany:$SLURM_JOB_ID my_second_job

Job dependencies documentation adapted from https://hpc.nih.gov/docs/userguide.html#depend