# -*- coding: UTF-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json

def reset_style():
	from matplotlib import rcParams
	rcParams['font.family'] = 'serif'
	rcParams['font.serif'] = ['Times New Roman']
	rcParams['axes.titlesize'] = 14
	rcParams['axes.labelsize'] = 14
	rcParams['lines.linewidth'] = 1.5
	rcParams['lines.markersize'] = 8
	rcParams['xtick.labelsize'] = 14
	rcParams['ytick.labelsize'] = 14
	rcParams['legend.fontsize'] = 14

reset_style()


def reset_plot(	width_in_inches = 4.5,
	height_in_inches = 4.5):
	dots_per_inch = 200
	plt.close()
	plt.figure(
		figsize=(width_in_inches, height_in_inches),
		dpi=dots_per_inch)


def rand_jitter(arr, scale=0.01):
    stdev = scale*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def heatscatter(x, y):
	heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	#plt.clf()
	reset_plot()
	plt.imshow(heatmap.T, extent=extent, origin='lower')
	plt.show()


def heatscatter_sns(x, y, figsize=(8,8)):
	sns.set(rc={'figure.figsize':figsize})
	sns.set(style="white", color_codes=True)
	sns.jointplot(x=x, y=y, kind='kde', color="skyblue")



def plot_training_history(history, par_dir):
	print(history.history.keys())

	# summarize history for r2
	try:
		plt.plot(history.history['r_squared'])
		plt.plot(history.history['val_r_squared'])
		plt.title('model r2')
		plt.ylabel('r2')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		#plt.show()
		plt.savefig(os.path.join(par_dir, 'r2.png'))
		plt.gcf().clear()
	except:
	 	pass

	# summarize history for loss

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()
	plt.savefig(os.path.join(par_dir, 'loss.png'))
	plt.gcf().clear()



def plot_controller_performance(controller_hist_file, metrics_dict, save_fn=None, N_sma=10):
	'''
	Example:
		controller_hist_file = 'train_history.csv'
		metrics_dict = {'acc': 0, 'loss': 1, 'knowledge': 2}
	'''
	#plt.clf()
	reset_plot()
	plt.grid(b=True, linestyle='--', linewidth=0.8)
	df = pd.read_csv(controller_hist_file, header=None)
	assert df.shape[0] > N_sma
	df.columns = ['trial', 'loss_and_metrics', 'reward'] + ['layer_%i'%i for i in range(df.shape[1]-3)] 
	#N_sma = 20
	
	plot_idx = []
	for metric in metrics_dict:
		metric_idx = metrics_dict[metric]
		df[metric] = [float(x.strip('\[\]').split(',')[ metric_idx ]) for x in df['loss_and_metrics'] ]
		df[metric+'_SMA'] = np.concatenate([[None]*(N_sma-1), np.convolve(df[metric], np.ones((N_sma,))/N_sma, mode='valid')])
		#df[metric+'_SMA'] /= np.max(df[metric+'_SMA'])
		plot_idx.append(metric+'_SMA')

	ax = sns.scatterplot(data=df[plot_idx])
	ax.set_xlabel('Steps')
	ax.set_ylabel('Simple Moving Average')
	if save_fn:
		plt.savefig(save_fn)
	else:
		plt.show()


def plot_environment_entropy(entropy_record, save_fn):
	'''plot the entropy change for the state-space
	in the training environment. A smaller entropy
	indicates converaged controller.
	'''
	#plt.clf()
	reset_plot()
	ax = sns.lineplot(np.arange(len(entropy_record)), entropy_record)
	ax.set_xlabel('Time step')
	ax.set_ylabel('Entropy')
	plt.savefig(save_fn)


def sma(data, window=10):
	return np.concatenate([np.cumsum(data[:window-1]) / np.arange(1,window), np.convolve(data, np.ones((window,))/window, mode='valid')])


def plot_action_weights(working_dir):
	save_path = os.path.join(working_dir, 'weight_data.json')
	if os.path.exists(save_path):
		df = json.load(open(save_path, 'r+'))
		#plt.clf()
		for layer in df:
			reset_plot(width_in_inches=6, height_in_inches=4.5)
			ax = plt.subplot(111)
			for type, data in df[layer].items():
				d = np.array(data)
				avg = np.apply_along_axis(np.mean, 0, d)
				ax.plot(avg, label=type)
				if d.shape[0] >= 6:
					std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
					min_, max_ = avg - 1.96*std, avg + 1.96*std
					ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

			box = ax.get_position()
			ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

			# Put a legend to the right of the current axis
			ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			ax.set_xlabel('Number of steps')
			ax.set_ylabel('Weight of layer type')
			#plt.title('Weight of Each Layer Type at Layer {}'.format(layer[-1]))
			plt.savefig(os.path.join(working_dir, 'weight_at_layer_{}.pdf'.format(layer[-1])))
	else:
		raise IOError('File does not exist')



def plot_stats(working_dir):
	save_path = os.path.join(working_dir, 'nas_training_stats.json')		
	if os.path.exists(save_path):
		df = json.load(open(save_path))
		#plt.clf()
		reset_plot()
		for item in ['Loss', 'Knowledge', 'Accuracy']:
			data = df[item]
			d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
			avg = np.apply_along_axis(np.mean, 0, d)
			plt.plot(avg, label=item)
			if d.shape[0]>=6:
				std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
				min_, max_ = avg - 1.96*std, avg + 1.96*std
				plt.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

		plt.legend(loc='best')
		plt.xlabel('Number of steps')
		plt.ylabel('Statistics')
		#plt.title('Knowledge, Accuracy, and Loss over time')
		plt.savefig(os.path.join(working_dir, 'nas_training_stats.pdf'))
	else:
		raise IOError('File not found')


def plot_stats2(working_dir):
	save_path = os.path.join(working_dir, 'nas_training_stats.json')		
	if os.path.exists(save_path):
		df = json.load(open(save_path))
		#plt.clf()
		reset_plot()
		#ax = plt.subplot(111)
		data = df['Loss']
		d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
		avg = np.apply_along_axis(np.mean, 0, d)
		ax = sns.lineplot(x=np.arange(1,len(avg)+1), y=avg, 
			color='b',label='Loss', legend=False)
		if d.shape[0]>=6:
			std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
			min_, max_ = avg - 1.96*std, avg + 1.96*std
			ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)
		
		ax2 = ax.twinx()
		data = df['Knowledge']
		d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
		avg = np.apply_along_axis(np.mean, 0, d)
		sns.lineplot(x=np.arange(1,len(avg)+1), y=avg, 
			color='g', label='Knowledge', ax=ax2, legend=False)
		if d.shape[0]>=6:
			std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
			min_, max_ = avg - 1.96*std, avg + 1.96*std
			ax2.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

		ax.figure.legend()
		ax.set_xlabel('Number of steps')
		ax.set_ylabel('Value')
		plt.savefig(os.path.join(working_dir, 'nas_training_stats.pdf'))
	else:
		raise IOError('File not found')


def accum_opt(data, find_min):
	tmp = []
	best = np.inf if find_min else -np.inf
	for d in data:
		if find_min and d < best:
				best = d
		elif (not find_min) and d > best:
				best = d  
		tmp.append(best)
	return tmp


def multi_distplot_sns(data, labels, save_fn, title='title', xlab='xlab', ylab='ylab', hist=False, rug=False, xlim=None, ylim=None, legend_off=False, **kwargs):
	assert len(data) == len(labels)
	#plt.clf()
	reset_plot()
	ax = sns.distplot(data[0], hist=hist, rug=rug, label=labels[0],  **kwargs)
	if len(data)>1:
		for i in range(1, len(data)):
			sns.distplot(data[i], hist=hist, rug=rug, label=labels[i], ax=ax, **kwargs)
	#ax.set_title(title, fontsize=16)
	ax.set_xlabel(xlab)
	ax.set_ylabel(ylab)
	if xlim:
		ax.set_xlim(xlim)
	if ylim:
		ax.set_ylim(ylim)
	ax.legend(loc='upper left')
	if legend_off:
		ax.get_legend().remove()
	if save_fn:
		plt.savefig(save_fn)
	else:
		return ax


def violin_sns(data, x, y, hue, save_fn=None, split=True, **kwargs):
	#plt.clf()
	reset_plot()
	ax = sns.violinplot(x=x, y=y, hue=hue, split=split, data=data, **kwargs)
	if save_fn:
		plt.savefig(save_fn)
	else:
		return ax