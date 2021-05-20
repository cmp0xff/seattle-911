
import numpy as np
import pandas as pd
from time import process_time as timer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import init
import prep
import model
import pred_te
from importlib import reload


def model_fwd_te(n_spi=2) :

    reload(init)
    reload(prep)
    reload(model)
    reload(pred_te)

    x, y = init.init()

    mod = model.model_ppl()

    print('Model parameters: ' + str(mod.get_params()))

    for tr_ind, te_ind in prep.fwd_splitter(n_spi=n_spi).split(x) :
        x_tr, y_tr = x.iloc[tr_ind], y.iloc[tr_ind]

        print('Training Gradient Boosting Regressor with .iloc[' + str(tr_ind[0]) + ':' + str(tr_ind[-1]) + ']')
        tim = timer()
        mod.fit(x_tr, y_tr)
        print('Training Gradient Boosting Regressor successful in ' + str(timer() - tim) + ' s')

        pred_te.printer(x, y, te_ind, mod)

def model_nai_te() :

    reload(init)
    reload(prep)
    reload(model)
    reload(pred_te)

    x, y = init.init()

    mod = model.model_ppl()
    
    print('Model parameters: ' + str(mod.get_params()))

    x, y = init.init()
    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    print('Training Gradient Boosting Regressor with .iloc[' + str(tr_ind[0]) + ':' + str(tr_ind[-1]) + ']')
    tim = timer()
    mod.fit(x.iloc[tr_ind], y.iloc[tr_ind])
    print('Training Gradient Boosting Regressor successful in ' + str(timer() - tim) + ' s')

    pred_te.printer(x, y, e1_ind, mod)
    pred_te.printer(x, y, e2_ind, mod)
