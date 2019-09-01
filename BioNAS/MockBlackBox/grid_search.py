# -*- coding: UTF-8 -*-
# given a state space, perform B times exhaustive searches
# zzj, 2.23.2019

import csv
import os
import itertools

def train_hist_csv_writter(writer, trial, loss_and_metrics, reward, model_states):
	data = [
		trial,
		[loss_and_metrics[x] for x in sorted(loss_and_metrics.keys())],
		reward
		]
	action_list = [str(x) for x in model_states]
	data.extend(action_list)
	writer.writerow(data)
	print(action_list)


def grid_search(state_space, manager, working_dir, B=10, resume_prev_run=True):
	write_mode = "a" if resume_prev_run else "w"
	fh = open(os.path.join(working_dir, 'train_history.csv'), write_mode)
	writer = csv.writer(fh)
	i = 0
	for b in range(B):
		for model_states in itertools.product(*state_space):
			i += 1
			print("B={} i={}".format(b, i))
			if not os.path.isdir(os.path.join(working_dir, 'weights', "trial_%i"%i)):
				os.makedirs(os.path.join(working_dir, 'weights', "trial_%i"%i))
			if resume_prev_run and os.path.isfile(os.path.join(working_dir, 'weights', "trial_%i"%i, "bestmodel.h5")):
				continue
			reward, loss_and_metrics = manager.get_rewards(trial=i, model_states=model_states)
			train_hist_csv_writter(writer, i, loss_and_metrics, reward, model_states)
			fh.flush()
	fh.close()
	return