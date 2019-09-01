for i in `seq -2 2`
do
	echo "qsub -N mock_$i ./qsub_mock.sh 100 $i tmp_mock_$i"
	qsub -N mock_$i ./qsub_mock.sh 100 $i tmp_mock_$i
	#bash ./qsub_mock.sh 6 $i tmp_mock_$i
	#break
done
