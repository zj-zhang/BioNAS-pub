''' make training labels for eCLIP model
by intersecting all peaks with 100bp genomic
bins.
Zijun Zhang
7.18.2018
'''

import re
from collections import defaultdict
#import pandas as pd


def compile_background_bins(gtf_fn='/u/nobackup/yxing/NOBACKUP/frankwoe/hg19/gencode.v19.annotation.gtf', binsize=100):
	BED_formatter = '{chrom}\t{start}\t{end}\t{peak_id}\t.\t{strand}\t{gene}\n'
	#fo = tempfile.NamedTemporaryFile()
	fo = open('gene_bins_%ibp.bed'%binsize, 'w')
	with open(gtf_fn, 'r') as fi:
		for line in fi:
			if line.startswith('#'):
				continue
			ele = line.strip().split('\t')
			if ele[2]!="gene":
				continue
			chrom = ele[0]
			start = int(ele[3])
			end = int(ele[4])
			strand = ele[6]
			gene = re.findall(r"(\w+)", ele[-1])[1].split('.')[0]
			for i in range(start, end, binsize):
				peak_id = ':'.join([chrom, str(i), str(i+binsize), strand])
				fo.write(BED_formatter.format(
					chrom=chrom, 
					start=i,
					end=i+binsize,
					peak_id=peak_id,
					gene=gene,
					strand=strand))
	fo.flush()
	fo.close()
	return


def bedtools_intersect():
	inter_cmd = '''
bedtools intersect -a gene_bins_100bp.bed \
-b filtered_peaks/*.bed \
-wa -wb -s -f 0.2 > gene_bins_100bp.with_peaks.bed
	'''
	os.system(inter_cmd)


def get_train_matrix(in_fn, out_fn):
	label_dict = defaultdict(lambda: defaultdict(int))
	#all_labels = [x.split('/')[-1].split('.')[0] for rbp in peak_dict for x in peak_dict]
	all_labels = set()
	with open(in_fn, 'r') as fi:
		for line in fi:
			ele = line.strip().split()
			id = ele[3]
			target = ele[11]
			label_dict[id][target] = 1
			all_labels.add(target)
	all_labels = sorted(list(all_labels))
	with open(out_fn, 'w') as fo:
		header = '\t'.join(['ID'] + all_labels) + '\n'
		fo.write(header)
		for id in label_dict:
			line = "{}\t{}\n".format(id, '\t'.join([str(label_dict[id][x]) for x in all_labels]) )
			fo.write(line)