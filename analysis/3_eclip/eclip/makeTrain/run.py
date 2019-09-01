
compile_background_bins(binsize=100)

peak_dict = filter_eclip_peaks(
	data_dir='/u/nobackup/yxing/NOBACKUP/frankwoe/DARTS-PNN/eclip/data', 
	out_dir='/u/nobackup/yxing/NOBACKUP/frankwoe/DARTS-PNN/eclip/filtered_peaks', 
	pval_cutoff=2.)


bedtools_intersect()

get_train_matrix('gene_bins_100bp.with_peaks.bed', 'label_matrix.gene_bins_100bp.with_peaks.bed')