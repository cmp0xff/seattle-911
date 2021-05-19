
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

import prep
import importlib
importlib.reload(prep)

def model_ppl(rseed=0) :
    return Pipeline([
        ('preprocessor', prep.prep_ppl()),
        ('regressor', GradientBoostingRegressor(random_state=rseed))
    ])
    
