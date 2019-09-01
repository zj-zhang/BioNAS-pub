import pandas as pd
import numpy as np
from ..utils.io import read_history
from ..Controller.state_space import get_layer_shortname
import scipy.stats as ss
from pkg_resources import resource_filename

pd.set_option('display.expand_frame_repr', False)
history_fn_list = [resource_filename('BioNAS.resources', 'mock_black_box/tmp_%i/train_history.csv'%i) for i in range(1,21)]


def ID2arch(hist_df, state_str_to_state_shortname):
	id2arch = {}
	for i in hist_df.ID:
		arch = tuple(state_str_to_state_shortname[x][hist_df.loc[hist_df.ID==i]['L%i'%(x+1)].iloc[0] ] for x in range(hist_df.shape[1]-5) )
		id2arch[i] = arch
	return id2arch


def get_gold_standard(history_fn_list, state_space):
	state_str_to_state_shortname = {}
	for i in range(len(state_space)):
		state_str_to_state_shortname[i] = {str(x):get_layer_shortname(x) for x in state_space[i]}
	df = read_history(history_fn_list)
	gs = df.groupby(by='ID', as_index=False).agg(np.median)
	gs['loss_rank'] = ss.rankdata(gs.loss)
	gs['knowledge_rank'] = ss.rankdata(gs.knowledge)
	
	id2arch = ID2arch(df, state_str_to_state_shortname)
	arch2id = {v:k for k,v in id2arch.items()}

	return gs, arch2id

