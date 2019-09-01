## this script starts up two asynchronous training processes and redirects stdout to two
## separate terminals while also dumping the output to two txt files
## usage: ./distributed_train.sh PTS_PORT_NUM  // where PTS_PORT_NUM is the other terminal whose pts number
## can be found by typing "tty" on the other terminal and check the number after /dev/pts/

#!/bin/sh
export CUDA_VISIBLE_DEVICES=-1
python ./BioNAS/train.py --job_name "ps" --task_index 0 &
export CUDA_VISIBLE_DEVICES=0
python ./BioNAS/train.py --job_name "worker" --task_index 0 | tee "$PWD"/BioNAS/terminal1.txt &
export CUDA_VISIBLE_DEVICES=3
python ./BioNAS/train.py --job_name "worker" --task_index 1 2>$1 | tee "$PWD"/BioNAS/terminal2.txt /dev/pts/"$1" &> /dev/null
