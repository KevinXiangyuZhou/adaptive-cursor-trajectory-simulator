Interactive Job - Using Slurm salloc commands
An interactive job is a job that returns a command line prompt (instead of running a script) when the job runs. Interactive jobs are useful when debugging or interacting with an application. Interactive jobs can also be useful in building Slurm batch scripts to run non-interactively. The salloc command is used to submit an interactive job to Slurm. When the job starts, a command line prompt will appear on one of the compute nodes assigned to the job. From here commands can be executed using the resources allocated on the local node.

Example

[user@gl-login1 ~]$ salloc --account=test
salloc: Pending job allocation 10081688
salloc: job 10081688 queued and waiting for resources
salloc: job 10081688 has been allocated resources
salloc: Granted job allocation 10081688
salloc: Waiting for resource configuration
salloc: Nodes gl3052 are ready for job
[user@gl3052 ~]$ hostname
gl3052.arc-ts.umich.edu
[user@gl3052 ~]$

Jobs submitted with salloc will be assigned the cluster default values of 1 CPU and 768MB of memory.  You should always specify the Slurm account to be used.  If additional resources are required, they can be requested as options to the salloc command. The following example job is assigned 2 nodes with 4 CPUS and 4GB of memory each:

Example

[user@gl-login1 ~]$ salloc --account=test --nodes=2 --ntasks-per-node=4 --mem-per-cpu=1GB --cpus-per-task=1
salloc: Pending job allocation 10081702
salloc: job 10081702 queued and waiting for resources
salloc: job 10081702 has been allocated resources
salloc: Granted job allocation 10081702
salloc: Nodes gl[3048-3049] are ready for job
[user@gl3048 ~]$ srun hostname
gl3048.arc-ts.umich.edu
gl3048.arc-ts.umich.edu
gl3048.arc-ts.umich.edu
gl3048.arc-ts.umich.edu
gl3049.arc-ts.umich.edu
gl3049.arc-ts.umich.edu
gl3049.arc-ts.umich.edu
gl3049.arc-ts.umich.edu
[user@gl3048 ~]$

In the above example, srun is used within the job from the first compute node to run a command once for every task in the job on the assigned resources. srun can be used to run on a subset of the resources assigned to the job. See the srun man page for more details.