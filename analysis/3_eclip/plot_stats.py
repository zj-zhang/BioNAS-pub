
from BioNAS.utils import io, plots

controller_hist_file = 'tmp_eclip/train_history.csv'

plots.plot_controller_performance(controller_hist_file, 
	metrics_dict={'acc':0}, save_fn='acc.pdf')

plots.plot_controller_performance(controller_hist_file, 
	metrics_dict={'loss':2}, save_fn='loss.pdf')

plots.plot_controller_performance(controller_hist_file, 
	metrics_dict={'knowledge':1}, save_fn='knowledge.pdf')