# -*- coding: UTF-8 -*-

import os
import shutil

from .. utils.plots import *

def post_processing_general(
            trial, 
            model, 
            hist, 
            data, 
            pred, 
            loss_and_metrics,
            working_dir='.',
            save_full_model=False
        ):
    par_dir = os.path.join(working_dir, 'weights', 'trial_%i'%trial)
    if os.path.isdir(par_dir):
        shutil.rmtree(par_dir)
    os.mkdir(par_dir)
    if save_full_model:
        model.save(os.path.join(working_dir, 'weights', 'trial_%i'%trial, 'full_bestmodel.h5'))
    plot_training_history(hist, par_dir)
    shutil.move(os.path.join(working_dir,'temp_network.h5'), os.path.join(par_dir, 'bestmodel.h5'))
    metadata = data[2] if len(data)>2 else None
    obs = data[1]
    write_pred_to_disk(
            os.path.join(par_dir, 'pred.txt'), 
            pred, obs, metadata, 
            loss_and_metrics
        )


def write_pred_to_disk(fn, y_pred, y_obs, metadata=None, metrics=None):
    with open(fn, 'w') as f:
        if metrics is not None:
            f.write( '\n'.join(['# {}: {}'.format(x, metrics[x]) for x in metrics]) + '\n' )
        f.write('pred\tobs\tmetadata\n')
        for i in range(len(y_pred)):
            if len(y_pred[i].shape)>1 or y_pred[i].shape[0]>1:
                y_pred_i = ','.join(['%.3f'%x for x in np.array(y_pred[i])])
                y_obs_i = ','.join(['%.3f'%x for x in np.array(y_obs[i])])
            else:
                y_pred_i = '%.3f'%y_pred[i]
                y_obs_i = '%.3f'%y_obs[i]                
            if metadata:
                f.write('%s\t%s\t%s\n'%(y_pred_i, y_obs_i, metadata[i]))
            else:
                f.write('%s\t%s\t%s\n'%(y_pred_i, y_obs_i, 'NA'))
    


def write_pred_to_disk_regression(fn, y_pred, y_obs, eid_list, metrics=None):
    with open(fn, 'w') as f:
        if metrics is not None:
            f.write( '\n'.join(['# {}: {}'.format(x, metrics[x]) for x in metrics]) + '\n' )
        f.write('pred\tobs\teid\n')
        for i in range(len(y_pred)):
            #f.write('%f\t%f\t%s\n'%(expit(y_pred[i]), expit(y_obs[i]), eid_list[i]))
            f.write('%f\t%f\t%s\n'%(y_pred[i], y_obs[i], eid_list[i]))
    
        os.system('Rscript plotPredScatter.R %s %s'% (fn, fn.rstrip('txt')+'pdf') )


def post_processing_regression(trial, model, train_hist, val_data, val_pred, val_metrics):
    pearson = val_metrics[-3]
    lins_con = val_metrics[-2]
    r2 = val_metrics[-1]
    _, y_val, eid_val, x_test, y_test, eid_test = val_data
    metrics = {'r2': r2, 'pearson': pearson, 'lins_con': lins_con}
    par_dir = os.path.join('weights', 'trial_%i'%trial)
    if os.path.isdir(par_dir):
        shutil.rmtree(par_dir)
    os.mkdir(par_dir)
    write_pred_to_disk_regression(os.path.join(par_dir, 'val.txt'), val_pred, y_val, eid_val, metrics)
    shutil.move('temp_network.h5', os.path.join(par_dir, 'bestmodel.h5'))
    plot_training_history(train_hist, par_dir)

    test_metrics_list = model.evaluate(x_test, y_test)
    y_test_pred = model.predict(x_test).flatten()
    test_metrics = {'r2': test_metrics_list[-1], 'pearson': test_metrics_list[-3],
        'lines_con': test_metrics_list[-2]}
    write_pred_to_disk_regression(os.path.join(par_dir, 'test.txt'), y_test_pred, y_test, eid_test, test_metrics)    

