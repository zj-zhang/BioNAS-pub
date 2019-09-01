for i in `seq 1 20`
do
	echo "qsub -N gs$i -q gpu.q -l GPU=1 ./qsub_grid_search.sh tmp_$i"
	qsub -N gs$i -q all.q ./qsub_grid_search.sh tmp_$i
done
