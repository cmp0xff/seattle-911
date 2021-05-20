
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
import sklearn.preprocessing as skl_prep
from sklearn.model_selection import TimeSeriesSplit

def fwd_splitter(n_spi=2, tr_week=261, te_week=52) :
    return TimeSeriesSplit(n_splits=n_spi, max_train_size=tr_week*7*24, test_size=te_week*7*24)

def nai_splitter() :
    e1_ind = -365*24
    tr_ind = (-365*5-1)*24+e1_ind
    e2_ind = (-365*5-1)*24+2*e1_ind
    return np.arange(tr_ind, e1_ind), np.arange(e1_ind, 0), np.arange(e2_ind, tr_ind)

def prep_ppl() :
    return(Pipeline([
        ('std_scl', skl_prep.StandardScaler())
    ]))
