# -*- coding: UTF-8 -*-

import os
import json
import numpy as np

import pandas as pd
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


width_in_inches = 4.5
height_in_inches = 4.5
dots_per_inch = 200

plt.figure(
    figsize=(width_in_inches, height_in_inches),
    dpi=dots_per_inch)
#plt.tight_layout()

def plot_action_weights_ax(working_dir, ax_set):
    save_path = os.path.join(working_dir, 'weight_data.json')
    if os.path.exists(save_path):
        df = json.load(open(save_path, 'r+'))
        i = 0
        for layer in df:
            ax = ax_set[i]
            i += 1
            for type, data in df[layer].items():
                if type.startswith('denovo'):
                    type = type.replace('denovo', 'regconv2d')
                d = np.array(data)
                assert d.shape[0] >= 6
                avg = np.apply_along_axis(np.mean, 0, d)
                std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                min_, max_ = avg - 1.96*std, avg + 1.96*std
                ax.plot(avg, label=type)
                ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

            ax.set_xlabel('Number of steps')
            ax.set_ylabel('Weight of layer type')
    else:
        raise IOError('File not found')
    return ax


def run():
    fig, axes = plt.subplots(4,4, figsize=(width_in_inches*4, height_in_inches*4), dpi=dots_per_inch)

    Lambda_list = [-3, 0, 3]
    for j in range(1,4):
        Lambda = Lambda_list[j-1]
        this_ax_set = axes[:,j]
        plot_action_weights_ax('tmp_mock_%i'%Lambda, this_ax_set)

    for i in range(4):
        handles, labels = axes[i,1].get_legend_handles_labels()
        axes[i,0].axis('off')
        axes[i,0].legend(handles, labels, loc='center')

    plt.savefig('diff_arch.pdf')


if __name__ == '__main__':
    run()