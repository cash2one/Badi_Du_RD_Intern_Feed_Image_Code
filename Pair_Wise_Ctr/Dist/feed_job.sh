#! /bin/bash
echo "==============JOB BEGIN============"
#/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun -npernode 1 feed_ps.sh
/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun feed_worker.sh
echo "===============JOB END============="
