BATCH FILE - Creating a Batch File
To be able to submit non-interactive jobs, you'll need to first prepare a batch script. This will allow you to run your workflows when resources become available and to submit multiple jobs at once. The batch script contains commands and directives for resource requirements (such as CPU, memory, and runtime length of your job) and  the instructions for running your program or workflow. It helps you request resources and run your job on an HPC cluster, making it easier to manage and organize your computational tasks.

A batch script is a file that's created using an editor, such as vim or emacs. It contains instructions that tell the scheduler the resources and steps for your job. If you are new to creating and editing files in the Linux environment, or Linux in general, visit: https://www.digitalocean.com/community/tutorials/an-introduction-to-linux-basics

Once you've prepared your batch script, you will submit it to the scheduler as a job.

A job is an allocation of resources assigned to an individual user for a specified amount of time. When a job runs, the scheduler selects and allocates resources to the job. The invocation of the application happens within the batch script, or at the command line for interactive and jobs. Job steps are sets of (possibly parallel) tasks within a job. 

Preparing your batch script
A batch job script is composed of three main components:

The interpreter used to execute the script
#SBATCH directives that convey Slurm scheduler options. (Note: a # in Slurm is not treated as a comment)
The application(s) to execute along with its input arguments and options
 

Let's look at an example batch script containing the previously mentioned components:

#!/bin/bash # COMMENT:The interpreter used to execute the script
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=example_job
#SBATCH --mail-user=uniqname@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=test
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:
<insert commands here>
 

Please note: if your job is set to utilize more than one node, make sure your code is MPI enabled in order to run across these nodes. You can use any of srun, mpirun, or mpiexec commands to start your MPI job.

For example, if you want to run a simple MPI job you would replace <insert commands here>with something like:

srun -N4 --mpi=pmix ./hello

The next section has more detail on these and other common job submission options one might include in their batch script.

For a more exhaustive list of options see Slurm's sbatch documentation. Also see Slurm's MPI documentation.

COMMON JOB SUBMISSION OPTIONS
Option	Slurm Command (#SBATCH)	Example Usage
Job name	--job-name=<name>	--job-name=gljob1
Account	--account=<account>	
--account=test

This should be set to the account that will pay for the work. In most cases this will be a UMRCP account, a researcher provider account, class account, or unit based account (LSA, COE, etc).

Queue	--partition=<name>	
--partition=standard

 

Available partitions: 

standard (default), gpu (GPU jobs only), largemem (large memory jobs only), debug, standard-oc (on-campus software only)

Wall time limit	--time=<dd-hh:mm:ss>	
--time=01-02:00:00

Typically, the MaxTime value on a partition will dictate how long your job can run. If you're unsure of how much time to ask for, you can test for up to 4 hours on the debug partition. Common pitfalls can be: not requesting enough wall time, resulting in your job terminating too soon, and requesting too much wall time which could delay the running of your job. For additional assistance, you can reach out to your unit support HPC staff.

Node count	--nodes=<count>	
--nodes=2

If not specified, the default behavior is to allocate enough nodes to  satisfy  the  requested resources  as  expressed  by  per-job  specification options (e.g. --ntasks, --cpus-per-tasks). For additional assistance determining your node geometry, you can reach out to your unit support HPC staff.

Process count per node	--ntasks-per-node=<count>	
--ntasks-per-node=1

Request that ntasks be invoked on each node. If you're doing MPI work, this will be relevant to you.

Core count (per process)	--cpus-per-task=<cores>	
--cpus-per-task=1

Without this option, Slurm will just try to allocate one processor per task.

Memory limit	--mem=<limit> (Memory per node in GiB, MiB, KiB)	--mem=12000m
If not set, the scheduler defaults will be applied.
Minimum memory per processor	--mem-per-cpu=<memory>	--mem-per-cpu=1000m
If not set, the scheduler defaults will be applied. This is the minimum memory required per usable allocated CPU. This will be governed, in part, by the amount of memory available on the node itself.  If you need more than 180G of memory per node, we suggest using the largemem partition.
Request GPUs	
--gres=gpu:<count>

--gpus=[type:]<number>

--gres=gpu:2

--gpus=2

Available types of GPUs: double precision v100, and single precision L40.
 

Process count per GPU	--ntasks-per-gpu=<count>
Must be used with --ntasks or --gres=gpu:	
--ntasks-per-gpu=2
--gres=gpu:4
 

2 tasks per GPU * 4 GPUs = 8 tasks total

Job array	--array=<array indices>	
--array=0-15

See section below for additional details on using arrays.

Standard output file	--output=<file path> (path must exist)	
--output=/home/%u/%x-%j.log

If not specified, Slurm combines ouput and error into a file named slurm-<jobId>.out. Slurm creates this file by default.

This file is potentially useful when troubleshooting your job.


Note:

%u = username
%x = job name
%j = job ID

Standard error file	--error=<file path> (path must exist)	
--error=/home/%u/error-%x-%j.log

If not specified, Slurm combines ouput and error into a file named slurm-<jobId>.out.

Copy environment	
--export=ALL (default)

--export=NONE (to not export environment)

--export=ALL
Copy environment variable	--export=<variable=value,var2=val2>	--export=EDITOR=/bin/vim
Job dependency	
--dependency=after:jobID[:jobID...]

--dependency=afterok:jobID[:jobID...]

--dependency=afternotok:jobID[:jobID...]

--dependency=afterany:jobID[:jobID...]

--dependency=after:1234[:1233]
Request software license(s)	
--licenses=<application>@slurmdb:<N>

Note: to see which software licenses are available run the following command from a login node: scontrol show lic
 

--licenses=stata@slurmdb:1
requests one license for Stata
Request event notification	
--mail-type=<events>

Note: multiple mail-type requests may be specified in a comma separated list:

--mail-type=BEGIN,END,NONE,FAIL,REQUEUE,ARRAY_TASKS

Note: there are multiple mail-type options one can use:

NONE, BEGIN, END, FAIL, REQUEUE, ALL, INVALID_DEPEND, STAGE_OUT, TIME_LIMIT, TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50, ARRAY_TASKS

--mail-type=BEGIN,END,FAIL
Email address	--mail-user=<email address>	--mail-user=uniqname@umich.edu
Defer job until the specified time	--begin=<date/time>	--begin=2020-12-25T12:30:00