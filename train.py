
import os
from sklearn.model_selection import GridSearchCV
from time import process_time as timer
import joblib

import init
import prep
import model

def train() :

    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')

    train_pt = './tmp/reg.pkl'
    if os.path.isfile(train_pt) :
        reg = joblib.load(train_pt)
        print('Regressor loaded from ' + train_pt)
    else:
        x, y = init.init()
        tr_ind, e1_ind, e2_ind = prep.nai_splitter()
    
        p_grid = {
            'regressor__learning_rate': [.5, .25, .1, .05, .01],
            'regressor__n_estimators': [25, 50, 100, 200, 400],
            'regressor__max_depth': [3, 4, 5]
        }

        tim = timer()
        reg = GridSearchCV(estimator=model.model_ppl(),
                           param_grid=p_grid,
                           n_jobs=-1,
                           verbose=2)

        print('Cross validating by Grid Search')
        reg.fit(x.iloc[tr_ind], y.iloc[tr_ind])
        print('Cross validating by Grid Search successful in ' + str(timer() - tim) + ' s')

        joblib.dump(reg, train_pt)
        print('Regressor dumped to ' + train_pt)
        
    return reg
