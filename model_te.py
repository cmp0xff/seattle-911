
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import init
import prep
import model
from importlib import reload

def model_fwd_te(n_spi=2) :

    reload(init)
    reload(prep)
    reload(model)

    x, y = init.init()

    mod = model.model_ppl()
    print(mod.get_params())

    for tr_ind, te_ind in prep.fwd_splitter(n_spi=n_spi).split(x) :
        x_tr, y_tr = x.iloc[tr_ind], y.iloc[tr_ind]
        x_te, y_te = x.iloc[te_ind], y.iloc[te_ind]

        mod.fit(x_tr, y_tr)

        y_pd = mod.predict(x_te)

        print(mod.score(x_te, y_te))

        print(np.sqrt(mean_squared_error(y_te, y_pd)))

        pd.DataFrame(data=y_pd, index=y_te.index, columns=['prediction']).join(y_te).resample('W').sum().plot()
        plt.show()

def model_nai_te() :

    reload(init)
    reload(prep)
    reload(model)

    x, y = init.init()

    mod = model.model_ppl()
    
    print(mod.get_params())

    x, y = init.init()
    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    mod.fit(x.iloc[tr_ind], y.iloc[tr_ind])

    y_p1 = mod.predict(x.iloc[e1_ind])
    print(mod.score(x.iloc[e1_ind], y.iloc[e1_ind]))
    print(np.sqrt(mean_squared_error(y.iloc[e1_ind], y_p1)))
    pd.DataFrame(data=y_p1, index=y.iloc[e1_ind].index, columns=['prediction']).join(y.iloc[e1_ind]).resample('W').sum().plot()
    plt.show()

    y_p2 = mod.predict(x.iloc[e2_ind])
    print(mod.score(x.iloc[e2_ind], y.iloc[e2_ind]))
    print(np.sqrt(mean_squared_error(y.iloc[e2_ind], y_p2)))
    pd.DataFrame(data=y_p2, index=y.iloc[e2_ind].index, columns=['prediction']).join(y.iloc[e2_ind]).resample('W').sum().plot()
    plt.show()
