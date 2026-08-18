OB OUTPUT - An Overview of Interpreting Job Outcomes
JOB OUTPUT
Recall the --output and --error batch file options from the primer. These are important to direct where your results go so that you can review them. If you do not, you may not be able to access them to troubleshoot your work.

By default, if these options are not specified, Slurm combines and saves them to a filename that includes the job ID (e.g. slurm-<jobId>.out). Slurm saves this file in the working directory from which the job was submitted. The files are written as soon as output is created.

For example if I submit job 93 from my home directory, the job output and error will be written to my home directory in a file called slurm-93.out. The file appears while the job is still running.

[user@gl-login1 ~]$ sbatch test.sh
Submitted batch job 93
[user@gl-login1 ~]$ ll slurm-93.out -rw-r–r– 1 user hpcstaff 122 Jun 7 15:28 slurm-93.out
[user@gl-login1 ~]$ squeue
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
93 standard example user R  0:04 1 gl3160

We suggest you save output to a networked filesystem, available on all login and compute nodes, like /home, /scratch, or /nfs.

If you submit your job from a working directory which is NOT a shared filesystem, (i.e. as in the previous example) your output will only be locally available on that node. For example, if I submit a job from /tmpon the login node, the output will be in /tmp on the compute node:

[user@gl-login1 tmp]$ pwd /tmp
[user@gl-login1 tmp]$ sbatch /home/user/test.sh
Submitted batch job 98
[user@gl-login1 tmp]$ squeue
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
98 standard example user R 0:03 1 gl3160
[user@gl-login1 tmp]$ ssh gl3160
[user@gl3160 ~]$ ll /tmp/slurm-98.out -rw-r–r– 1 user hpcstaff 78 Jun 7 15:46 /tmp/slurm-98.out

NOTE: We caution against saving to local filesystems. Locally saved data will need to be copied to another location after the job completes (either manually, or by way of action defined in your batch script if saving to disk is absolutely necessary). Slurm is configured to only allow ssh to a compute node if the user has a running job on it. If your job is no longer running, you no longer have the ability to ssh to that node and get your output back.

ACCOUNTING AND JOB STATISTICS
Accounting
Knowing how much an account has used is key to being able to submit work reliably. ARC provides a number of options to help find this information.

The Research Management Portal (RMP) is a great way to gain insight into your account utilization.

From the command line, ARC offers a script called my_account_usage help users report on the monthly cost of an account. Here's what the script can provide:

[user@gl-login1 ~]$ my_account_usage -h
usage: my_account_usage -A ACCOUNT [-Y YEAR]

Report or estimate the monthly cost for a given account

optional arguments:
  -h, --help        show this help message and exit
  -A , --account    The account to report
  -Y , --year       The fiscal year
  -S , --start      The start year-month as numbers, example: 2020-01
  -E , --end        The end year-month as numbers, example: 2021-01
  -p, --percentage  Print perentages for each user
  -d, --debug       Debug modes. Print raw Slurm outputs.
  -1, --sort1       Sort by user total for range
  -2, --sort2       Sort by user total for this month
  -3, --sort3       Sort by user total for last month

Job Statistics
Understanding how your job ran is important. Viewing a jobs statistics is a great way for a user to see if the resources they're requesting are being utilized. It provides an opportunity to assess and optimize your job requirements, which can lead to have your jobs starting faster and costing less.

One way to view job statistics us on the command line using the ARC-provided utility my_job_statistics. You simply pass the job ID of the job statistics you wish to view: my_job_statistics -j <job_ID>

Another great way to get job statistics is from the job completion emails. These contain helpful tips based on your job run, and are sent automatically at job completion time. Make sure you don't have any specific #SBATCH overrides barring emails from being sent.