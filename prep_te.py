
import init
import prep
from importlib import reload

def fwe_te() :

    reload(init)
    reload(prep)

    x, y = init.init()
    for tr_ind, te_ind in prep.fwd_splitter().split(x) :
        print(x.iloc[tr_ind].info(), y.iloc[te_ind].head(2), len(y))

def nai_te() :

    reload(init)
    reload(prep)

    x, y = init.init()
    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    print(tr_ind, e1_ind, e2_ind)
    print(x.iloc[tr_ind].info(), y.iloc[e1_ind].head(2), y.iloc[e2_ind].head(2),)
