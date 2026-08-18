Users interact with an HPC cluster through login nodes. When using ssh via the command line to access any of our clusters you will be placed on a login node.

Login nodes are a place where users can login, edit files, view job results and submit new jobs.
Login nodes are a shared resource and should not be used to run application workloads.
There are limits on the login nodes.
These measures are intended to allow users to interact with the Slurm environment, transfer data, and perform basic testing, while preventing any single user from monopolizing the entire login node for testing purposes.
This enhances the overall usability of the environment for everyone.
If you require more resources for testing, submit an interactive job or use Open OnDemand in the debug queue. 