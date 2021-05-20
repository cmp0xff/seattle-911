
import init
from importlib import reload

def calls_te() :

    reload(init)

    calls_df = init.init_calls()
    print(calls_df.info())
    print(calls_df.head(1))
    # print(calls_df.tail(2))
    # print(calls_df.iloc[calls_df.index.duplicated()])

def weather_te() :

    reload(init)

    wtr_df = init.init_weather()
    print(wtr_df.info())
    print(wtr_df.head(1))
    # print(wtr_df.iloc[wtr_df.index.duplicated(keep=False)])

def init_te() :

    reload(init)

    x, y = init.init()
    print(x.info())
    print(x.head(1))
    print(y.head(1))
    # print(x.iloc[x.index.duplicated(keep=False)])
