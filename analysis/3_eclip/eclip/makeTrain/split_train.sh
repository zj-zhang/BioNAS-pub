
grep chr8 label_matrix.gene_bins_100bp.with_peaks.bed > label_matrix.testing_chr8.bed
grep -v chr8 label_matrix.gene_bins_100bp.with_peaks.bed | shuf > tmp
head -n 147531 tmp > label_matrix.val.bed
tail -n +147532 tmp > label_matrix.train.bed
gzip label_matrix.*.bed