#!/bin/bash
#$ -l h_vmem=16G
#$ -l m_mem_free=16G
#$ -R y
#$ -V
#$ -cwd
#$ -j y
#$ -m a
#$ -M zzj.zju@gmail.com

python examples/grid_search.simple_conv1d_state_space.py $1
