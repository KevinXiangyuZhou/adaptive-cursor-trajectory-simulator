SINFO
A cluster is often highly utilized and may not be able to run a job when it is submitted. When this occurs, the job is placed in a partition. Specific compute node resources are defined for every job partition. The Slurm partition is synonymous with the term queue.

Each partition can be configured with a set of limits which specify the requirements for every job that can run in that partition. These limits include job size, wall clock limits, and the users who are allowed to run in that partition.

Here are some partition examples from the Great Lakes cluster:

"standard" (used for most production jobs, 14 day max walltime)
"largemem" (used for jobs that require large amounts of RAM, 14 day max)
"gpu" (used for GPU-intensive tasks, 14 day max)
"debug" (only to verify/debug jobs, 4 hour max)
"viz" (used for visualization jobs, 4 hour max)
"viz-long" (used for visualization jobs, 48 hour max)
Commands related to partitions include:

sinfo	Lists all partitions currently configured
scontrol show partition
 	Provides details about all partitions
scontrol show partition <name>	Provide details about a specific partition
squeue	Lists all jobs currently on the system, one line per job
The sinfo command displays information about the state of partitions, and nodes. When you run sinfo by itself, it provides a quick overview of the cluster's current state, including which partitions are available, what their status is, how many nodes they contain, and how those nodes are allocated.

Here is a more granular example, using format options, to show how you would run sinfo to get a summary of cluster resources on a per-partition basis:

$ sinfo -o "%P|%a|%D|%C|%m|%d|%T|%N"

Explanation of the format options used:

%P - Partition name
%a - Availability of the partition, yes or no
%D - Total number of nodes in the partition
%C - Total number of CPUs allocated, idle, other, and total
%m - Memory size of the partition
%d - Disk space available on the partition (if applicable)
%T - State of the nodes (allocated, idle, mixed, etc.)
%N - Names of nodes in the partition
Each specifier provides a different piece of information about the partition, and by combining them, you get a formatted output that is easy to read and interpret.

Here's an interpretation of some possible %T (States) you could see:

allocated - All CPUs on the node are allocated to jobs.
idle - No jobs are currently running on the node; it's available.
mixed - Some CPUs on the node are allocated, but not all.
down - The node is not operational.
Remember that the output format can be customized to show different details as per your requirements, and you can consult the Slurm documentation or use man sinfo for more information on the available options.

Before running this command or any of the Slurm commands (scontrol show node or scontrol show partition), make sure you are logged into the system where Slurm is installed and configured (e.g. a login node).