Submit a Single Job - Slurm sbatch
The sbatch command is used to submit a batch script to Slurm. Slurm will reject the job at submission time if there are requests or constraints that Slurm cannot fulfill as specified. This gives the user the opportunity to examine the job request and resubmit it with the necessary corrections.

To submit a batch script, while logged into the cluster (i.e. ssh to the cluster and be on a login node), simply run: sbatch <jobScriptFile>

Below are additional job commands one might run.

Common Job Commands
Command	Slurm
Submit a job	sbatch <job script>
Delete a job	scancel <job ID>
Job status (all)	squeue
Job status (by job)	squeue -j <job ID>
Job status (by user)	sq        (equivalent of: squeue -u <your_uniqname>)
sq <user> (equivalent of: squeue -u <user>)

Job status (detailed)	scontrol show job -dd <job ID>
Show expected start time	squeue -j <job ID> --start
Queue list / info	scontrol show partition <name>
Node list	scontrol show nodes
Node details	scontrol show <node>
Hold a job	scontrol hold <job ID>
Release a job	scontrol release <job ID>
Cluster status	sinfo
Start an interactive job	salloc <args>
X forwarding	salloc <args> --x11
Read stdout messages at runtime	No equivalent command / not needed. Use the --output option instead.
Monitor or review a job’s resource usage	sacct -j <job_num> --format JobID,jobname,NTasks,nodelist,CPUTime,ReqMem,Elapsed
(see sacct for all format options)

View job batch script	scontrol write batch_script <jobID>
 
View accounts you can submit to	sacctmgr show assoc user=$USER
View users with access to an account	sacctmgr show assoc account=<account>
View default submission account and wckey	sacctmgr show user <account>