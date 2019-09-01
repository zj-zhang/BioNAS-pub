#!/bin/bash
#$ -l h_vmem=16G
#$ -l m_mem_free=16G
#$ -R y
#$ -V
#$ -cwd
#$ -j y
#$ -m be
#$ -M zzj.zju@gmail.com

python multitask_eclip.py
