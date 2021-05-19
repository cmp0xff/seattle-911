
import pandas as pd
from sklearn.model_selection import train_test_split

import init

def train_test(tsize=.3, rseed=0) :
    xy_all = init.init()
    xy_df = xy_all.loc['2016':'2020']

    return train_test_split(
        xy_df.iloc[:, 1:], 
        xy_df.iloc[:, 0], 
        test_size=tsize,
        random_state=rseed) # + (xy_all.loc['2015'].iloc[:, 1:], xy_all.loc['2015'].iloc[:, 0])

from sklearn.pipeline import Pipeline
import sklearn.preprocessing as skl_prep

def prep_ppl() :
    return(Pipeline([
        ('std_scl', skl_prep.StandardScaler())
    ]))
