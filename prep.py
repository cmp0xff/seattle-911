
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def fwd_splitter(n_spi=2, tr_week=261, te_week=52) :
    return TimeSeriesSplit(n_splits=n_spi, max_train_size=tr_week*7*24, test_size=te_week*7*24)

import numpy as np

def nai_splitter() :
    te_ind = -365*24
    tr_ind = (-365*5-1)*24+te_ind-1
    return np.arange(tr_ind, te_ind), np.arange(te_ind, -1)

from sklearn.pipeline import Pipeline
import sklearn.preprocessing as skl_prep

def prep_ppl() :
    return(Pipeline([
        ('std_scl', skl_prep.StandardScaler())
    ]))
