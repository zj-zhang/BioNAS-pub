#!/bin/bash
#$ -l h_vmem=8G
#$ -l m_mem_free=8G
#$ -R y
#$ -V
#$ -cwd
#$ -j y
#$ -m be
#$ -M zzj.zju@gmail.com

python mock_func_caller.py $1 $2 $3 $4
