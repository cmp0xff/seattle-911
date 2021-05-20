
from time import process_time as timer

import init
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
