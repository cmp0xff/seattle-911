
import os
from sklearn.model_selection import GridSearchCV
import joblib

import init
import prep
import model

def train() :

    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    train_pt = './tmp/reg.pkl'
    if os.path.isfile(train_pt) :
        print('Load saved regressor from ' + train_pt)
        reg = joblib.load(train_pt)
    else:
        x, y = init.init()
        tr_ind, e1_ind, e2_ind = prep.nai_splitter()
    
        p_grid = {
            'regressor__n_estimators': [100, 300, 500],
            'regressor__max_depth': [3, 4, 5]
        }

        reg = GridSearchCV(estimator=model.model_ppl(),
                           param_grid=p_grid,
                           n_jobs=-1,
                           verbose=2)

        reg.fit(x.iloc[tr_ind], y.iloc[tr_ind])
        
    return reg
