#! /bin/bash
echo "==============JOB BEGIN============"
#chown slurm:slurm hdp.sh
#sh hdp.sh

/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun --bind-to none worker.sh >./log/tmp

#/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun worker.sh >./log/tmp
echo "===============JOB END============="
