#! /bin/bash
echo "==============JOB BEGIN============"
/home/HGCP_Program/software-install/openmpi-1.8.5/bin/mpirun --bind-to none worker.sh >./log/tmp
echo "===============JOB END============="
