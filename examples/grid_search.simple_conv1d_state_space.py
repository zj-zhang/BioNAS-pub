# -*- coding: UTF-8 -*-

from simple_conv1d_state_space import *
from BioNAS.MockBlackBox import grid_search
import sys

def main(wd):
    state_space = get_state_space()
    # then build reward function from K_function
    mkf = get_motif_Kfunc()
    reward_fn = get_reward_func(mkf)

    # set alias for model_fn
    model_fn = build_sequential_model

    # read the data
    train_data, validation_data = get_data()

    # init network manager
    manager = get_manager(train_data, validation_data, model_fn, reward_fn, wd)

    # grid search
    wd = './tmp' if wd is None else wd
    grid_search.grid_search(state_space, manager, wd, B=1)


if __name__ == '__main__':
    try:
        wd = sys.argv[1]
    except:
        wd = None
    print("workding_dir = " + str(wd))
    if not os.path.isdir(wd):
        os.makedirs(wd)
    main(wd)