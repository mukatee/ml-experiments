#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No output file provided. Usage: ./log_gpu_cpu_mem.sh /path/to/output.csv"
    exit 1
fi

FILE=$1

# Write header to file
echo "timestamp,epoch_seconds,cpu_usage [%],memory.used [MiB],memory.total [MiB],gpu_utilization [%],gpu_memory.used [MiB],gpu_memory.total [MiB]" > $FILE

while true; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
    EPOCH=$(date +%s)
    CPU_USAGE=$(vmstat 1 2 | tail -1 | awk '{print $15}')
    MEMORY_USAGE=$(free -m | awk 'NR==2{printf "%s", $3}')
    MEMORY_TOTAL=$(free -m | awk 'NR==2{printf "%s", $2 }')
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    echo "${TIMESTAMP},${EPOCH},${CPU_USAGE},${MEMORY_USAGE},${MEMORY_TOTAL},${GPU_STATS}" >> $FILE
    sleep 1
done

