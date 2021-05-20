
from time import process_time as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import init, prep, model, train, main
from importlib import reload

def calls_te() :

    reload(init)
    
    print('Testing calls parsing from given database')

    tim = timer()
    calls_df = init.init_calls()
    print(calls_df.info())
    print(calls_df.head(1))
    # print(calls_df.tail(2))
    # print(calls_df.iloc[calls_df.index.duplicated()])
    
    print('Testing calls parsing from given database successful in '+ str(timer() - tim) + ' s')

def weather_te() :

    reload(init)

    print('Testing weather parsing from given database')
    
    tim = timer()
    wtr_df = init.init_weather()
    print(wtr_df.info())
    print(wtr_df.head(1))
    print(wtr_df.tail(1))
    # print(wtr_df.iloc[wtr_df.index.duplicated(keep=False)])
    
    print('Testing weather parsing from given database successful in '+ str(timer() - tim) + ' s')

def init_te() :

    reload(init)
    
    print('Testing database initialisation from given databases')

    tim = timer()
    x, y = init.init()
    print(x.info())
    print(x.head(1))
    print(y.head(1))
    # print(x.iloc[x.index.duplicated(keep=False)])
    
    print('Testing database initialisation from given databases successful in '+ str(timer() - tim) + ' s')

def fwe_te() :

    reload(init)
    reload(prep)

    x, y = init.init()
    for tr_ind, te_ind in prep.fwd_splitter().split(x) :
        print(x.iloc[tr_ind].info(), y.iloc[te_ind].head(2), len(y))

def nai_te() :

    reload(init)
    reload(prep)

    x, y = init.init()
    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    print(tr_ind, e1_ind, e2_ind)
    print(x.iloc[tr_ind].info(), y.iloc[e1_ind].head(2), y.iloc[e2_ind].head(2),)

def printer(x, y, te_ind, mod, binn='W') :
    print('Testing Gradient Boosting Regressor with .iloc[' + str(te_ind[0]) + ', ' + str(te_ind[-1]) + ']')

    y_pd = mod.predict(x.iloc[te_ind])

    print('Score: ' + str(mod.score(x.iloc[te_ind], y.iloc[te_ind])))
    print('R^2: ' + str(np.sqrt(mean_squared_error(y.iloc[te_ind], y_pd))))

    pd.DataFrame(data=y_pd, 
                 index=y.iloc[te_ind].index, 
                 columns=['prediction']).join(y.iloc[te_ind]).resample(binn).sum().plot()
    plt.title('Gradient Boosting Regressor with .iloc[' + str(te_ind[0]) + ':' + str(te_ind[-1]) + ']')
    plt.show()

def model_fwd_te(n_spi=2) :

    reload(init)
    reload(prep)
    reload(model)

    x, y = init.init()

    mod = model.model_ppl()

    print('Model parameters: ' + str(mod.get_params()))

    for tr_ind, te_ind in prep.fwd_splitter(n_spi=n_spi).split(x) :
        x_tr, y_tr = x.iloc[tr_ind], y.iloc[tr_ind]

        print('Training Gradient Boosting Regressor with .iloc[' + str(tr_ind[0]) + ':' + str(tr_ind[-1]) + ']')
        tim = timer()
        mod.fit(x_tr, y_tr)
        print('Training Gradient Boosting Regressor successful in ' + str(timer() - tim) + ' s')

        printer(x, y, te_ind, mod)

    
def model_nai_te() :

    reload(init)
    reload(prep)
    reload(model)

    x, y = init.init()

    mod = model.model_ppl()
    
    print('Model parameters: ' + str(mod.get_params()))

    x, y = init.init()
    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    print('Training Gradient Boosting Regressor with .iloc[' + str(tr_ind[0]) + ':' + str(tr_ind[-1]) + ']')
    tim = timer()
    mod.fit(x.iloc[tr_ind], y.iloc[tr_ind])
    print('Training Gradient Boosting Regressor successful in ' + str(timer() - tim) + ' s')

    printer(x, y, e1_ind, mod)
    printer(x, y, e2_ind, mod)

def train_te(retrain=False) :

    reload(init)
    reload(prep)
    reload(train)

    x, y = init.init()

    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    reg = train.train(redo=retrain)

    printer(x, y, e1_ind, reg)
    printer(x, y, e2_ind, reg)

import os
import pandas as pd

import init, main

def data_gen(fwtr='./data/test_in.csv', fcalls='./data/test_cal.csv') :
    wtr_dr = init.weather_parser()

    calls_dr = init.calls_parser()
    calls_dr = calls_dr.loc['2020-07-04']
    calls_dr.index.rename('Datetime', inplace=True)
    calls_dr.index = calls_dr.index.strftime("%m/%d/%Y %I:%M:%S %p")
    print(calls_dr.head())

    wtr_dr.loc['2020-07-04'].to_csv(fwtr)
    calls_dr.to_csv(fcalls)
