Managing Jobs - squeue and scontrol (active job status)
Job Status
The quickest way to see your job status is with the command: sq. This a command is shorthand for squeue -u <your_uniqname>, which you can also run. If you wish to see the status of a specific job id, then you can run: squeue -j <job_ID>

Another way to get a job’s specifications can be seen by invoking scontrol show job <jobID>.  More details about the job can be written to a file by using scontrol write batch_script <jobID> output.txt. If no output file is specified, the script will be written to slurm-<jobID>.sh.

Slurm captures and reports the exit code of the job script (sbatch jobs) as well as the signal that caused the job’s termination when a signal caused a job’s termination.

NOTE: A job’s record remains in Slurm’s memory for 30 minutes after it completes.  scontrol show job will return “Invalid job id specified” for a job that completed more than 30 minutes ago.  At that point, one must invoke the sacct command to retrieve the job’s record from the Slurm database.

Modifying a Batch Job
Many of the batch job specifications can be modified after a batch job is submitted and before it runs.  Typical fields that can be modified include the job size (number of nodes), partition (queue), and wall clock limit.  Job specifications cannot be modified by the user once the job enters the Running state.

Beside displaying a job’s specifications, the scontrol command is used to modify them.  Examples:

scontrol -dd show job <jobID>	Displays all of a job’s characteristics
scontrol write batch_script <jobID>	Retrieve the batch script for a given job
scontrol update JobId=<jobID> Account=science	Change the job’s account to the “science” account
scontrol update JobId=<jobID> Partition=standard	Changes the job’s partition to the priority partition
Holding and Releasing a Batch Job
If a user’s job is in the pending state waiting to be scheduled, the user can prevent the job from being scheduled by invoking the scontrol hold <jobID> command to place the job into a Held state. Jobs in the held state do not accrue any job priority based on queue wait time.  Once the user is ready for the job to become a candidate for scheduling once again, they can release the job using the scontrol release <jobID> command.

Canceling and Signaling a Batch Job
Both Running and Pending jobs can be cancelled (withdrawn from the queue) using the scancel command (scancel <jobID>).    If the job is Running, the default behavior is to issue the job a SIGTERM, wait 30 seconds, and if processes from the job continue to run, issue a SIGKILL command.

The -s option of the scancel command (scancel -s <signal> <jobID>) allows the user to issue any signal to a running job.

Job States
The basic job states are these:

Pending – the job is in the queue, waiting to be scheduled
Held – the job was submitted, but was put in the held state (ineligible to run)
Running – the job has been granted an allocation.  If it’s a batch job, the batch script has been run
Complete – the job has completed successfully
Timeout – the job was terminated for running longer than its wall clock limit
Preempted – the running job was terminated to reassign its resources to a higher QoS job
Failed – the job terminated with a non-zero status
Node Fail – the job terminated after a compute node reported a problem
None -- If there are more than 20 jobs, you may get this job state before it becomes available to schedule. [the backfill scheduler  evaluates jobs in priority order, so when the backfill scheduler hits a the max number of jobs to evaluate , it stops evaluating and gives the reason None.  In the case of Great Lakes, that number is currently 20. ]
For the complete list, see the “JOB STATE CODES” section under the squeue man page.

Pending Reasons
A pending job can remain pending for a number of reasons:

Dependency – the pending job is waiting for another job to complete
Priority – the job is not high enough in the queue
Resources – the job is high in the queue, but there are not enough resources to satisfy the job’s request
Partition Down – the queue is currently closed to running any new jobs
For the complete list, see the “JOB REASON CODES” section under the squeue man page.

Job Checkpointing to Avoid Lost Work Due to Walltime Overruns
If a running application overruns its wall clock limit, all its work could be lost. To prevent such an outcome, applications have two means for discovering the time remaining in the application.

The first means is to use the sbatch --signal=<sig_num>[@<sig_time>] option to request a signal (like USR1 or USR2) at sig_time number of seconds before the allocation expires. The application must register a signal handler for the requested signal in order to to receive it. The handler takes the necessary steps to write a checkpoint file and terminate gracefully.
The second means is for the application to issue a library call to retrieve its remaining time periodically. When the library call returns a remaining time below a certain threshold, the application can take the necessary steps to write a checkpoint file and terminate gracefully.
Slurm offers the slurm_get_rem_time() library call that returns the time remaining. On some systems, the yogrt library (man yogrt) is also available to provide the time remaining.