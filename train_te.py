
import pandas as pd
from time import process_time as timer
import matplotlib.pyplot as plt

import init
import prep
import train
import pred_te
from importlib import reload

def train_te() :

    reload(init)
    reload(prep)
    reload(train)
    reload(pred_te)

    x, y = init.init()

    tr_ind, e1_ind, e2_ind = prep.nai_splitter()

    reg = train.train()

    pred_te.printer(x, y, e1_ind, reg)
    pred_te.printer(x, y, e2_ind, reg)
