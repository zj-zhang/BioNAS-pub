# -*- coding: UTF-8 -*-

from BioNAS.MockBlackBox.gold_standard import get_gold_standard, history_fn_list
from BioNAS.MockBlackBox.simple_conv1d_space import get_state_space
from BioNAS.utils.io import read_action_weights
from BioNAS.utils.plots import plot_stats, plot_action_weights

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


def read_data_and_plot():
	# prepare gold standard
    state_space = get_state_space()
    gs, arch2id = get_gold_standard(history_fn_list, state_space)
    gs.index = gs.ID

    # read in converged architecture for each run
    Lambda_list = [-3, -2, -1, 0, 1, 2, 3]
    gs_list = []
    for Lambda in Lambda_list:
        archs = read_action_weights("tmp_mock_%i/weight_data.json"%Lambda)
        this_gs = gs.loc[ [arch2id[x] for x in archs] ]
        this_gs['Lambda'] = Lambda
        gs_list.append(this_gs)

    eval_df = pd.concat(gs_list)
    eval_df.index = zip(eval_df.Lambda, eval_df.ID)

    # plot gold standard vs lambda
    plt.clf()
    plot_df = pd.melt(eval_df, id_vars=['Lambda'], value_vars=['loss_rank', 'knowledge_rank'])
    ax = sns.lineplot(x='Lambda', y='value', 
        hue='variable', style='variable', 
        data=plot_df, 
        ci=95,
        markers=True, dashes=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=['Loss', 'Knowledge'], loc='upper left')    
    ax.set_xlabel('log10(Lambda)')
    ax.set_ylabel('Rank')
    plt.savefig('rank.pdf')

    plt.clf()
    plot_df = pd.melt(eval_df, id_vars=['Lambda'], value_vars=['loss', 'knowledge'])
    ax = sns.lineplot(x='Lambda', y='value', 
        hue='variable', style='variable', 
        data=plot_df, 
        ci=95,
        markers=True, dashes=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=['Loss', 'Knowledge'], loc='upper right')    
    
    ax.set_xlabel('log10(Lambda)')
    ax.set_ylabel('Value')
    plt.savefig('value.pdf')

    # plot best lambda nas
    for Lambda in Lambda_list:
        plot_stats('tmp_mock_%i'%Lambda)
        plot_action_weights('tmp_mock_%i'%Lambda)


def main():
    read_data_and_plot()

if __name__ == '__main__':
    main()