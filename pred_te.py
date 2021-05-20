
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def printer(x, y, te_ind, mod) :
    print('Testing Gradient Boosting Regressor with .iloc[' + str(te_ind[0]) + ', ' + str(te_ind[-1]) + ']')

    y_pd = mod.predict(x.iloc[te_ind])

    print('Score: ' + str(mod.score(x.iloc[te_ind], y.iloc[te_ind])))
    print('R^2: ' + str(np.sqrt(mean_squared_error(y.iloc[te_ind], y_pd))))

    pd.DataFrame(data=y_pd, 
                 index=y.iloc[te_ind].index, 
                 columns=['prediction']).join(y.iloc[te_ind]).resample('W').sum().plot()
    plt.title('Gradient Boosting Regressor with .iloc[' + str(te_ind[0]) + ':' + str(te_ind[-1]) + ']')
    plt.show()
