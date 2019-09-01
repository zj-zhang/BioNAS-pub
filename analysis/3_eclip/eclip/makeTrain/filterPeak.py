'''iterate through all files to 
filter out non-enriched eclip peaks
Zijun Zhang
7.18.2018
'''

import os
import gzip
from collections import defaultdict

def filter_eclip_peaks(data_dir, out_dir, pval_cutoff):
	peak_dict = defaultdict(list)
	peak_files = [os.path.join(data_dir, x) 
		for x in os.listdir(data_dir) if x.endswith('.bed.gz') ]
	for peak_fn in peak_files:
		with gzip.open(peak_fn) as f:
			line = f.readline()
			ele = line.strip().split('\t')
			pval = float(ele[7])
			try:
				rbp, cell, replicate = ele[3].split('_')
			except:
				print line, ele[3]
				continue
			new_fn = ele[3] + '.bed'
			peak_dict[rbp+'_'+cell].append(new_fn)
			fo = open(os.path.join(out_dir, new_fn), 'w')
			if pval > pval_cutoff:
				fo.write(line)
			for line in f:
				ele = line.strip().split()
				pval = float(ele[7])
				if pval > pval_cutoff:
					fo.write(line)
			fo.close()
	return peak_dict