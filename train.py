
import os
from sklearn.model_selection import GridSearchCV
from time import process_time as timer
import joblib

import init, prep, model

def train(redo = False) :

    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')

    train_pt = './tmp/reg.pkl'
    if not redo and os.path.isfile(train_pt) :
        reg = joblib.load(train_pt)
        print('Regressor loaded from ' + train_pt)
    else:
        x, y = init.init()
        tr_ind, e1_ind, e2_ind = prep.nai_splitter()
    
        p_grid = {
            'regressor__learning_rate': [.1,], # .1
            'regressor__n_estimators': [x + 100 for x in range(5)], # 100
            'regressor__subsample': [1.], # 1.
            'regressor__max_depth': [3], # 3
            'regressor__max_features': ['auto'],
            'regressor__tol': [1e-4],
        }

        tim = timer()
        reg = GridSearchCV(estimator=model.model_ppl(),
                           param_grid=p_grid,
                           n_jobs=-1,
                           verbose=2)

        print('Cross validating by Grid Search')
        reg.fit(x.iloc[tr_ind], y.iloc[tr_ind])
        # print('Cross validating by Grid Search successful in ' + str(timer() - tim) + ' s')
        
        print(reg.best_params_)

        joblib.dump(reg, train_pt)
        print('Regressor dumped to ' + train_pt)
        
    return reg
