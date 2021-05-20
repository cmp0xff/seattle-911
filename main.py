
import sys, getopt
import numpy as np
import pandas as pd

import init, train, tester

def usage() :
    print('usage: python ' + sys.argv[0] + ' -w weather.csv [-c calls.csv]')

def main() :
    try :
        opts, args = getopt.getopt(sys.argv[1:], 'hw:c:o:', ['help', 'weather=', 'calls=', 'output='])
    except getopt.GetoptError as err :
        print(err)
        usage()
        sys.exit(2)

    fwtr = ''
    fcal = ''
    fout = './output.csv'

    for o, a in opts :
        if o in ('-h', '--help') :
            usage()
            sys.exit()
        elif o in ('-w', '--weather') :
            fwtr = a
        elif o in ('-c', '--calls') :
            fcal = a
        elif o in ('-o', '--output') :
            fout = a
    
    if not (fwtr and sys.isfile(fwtr)) :
        fwtr = './tmp/test_wtr.csv'
        fcal = './tmp/test_cal.csv'
        print('No weather database. Work in demo mode with ' + fwtr + ' and ' + fcal)
        tester.data_gen(fwtr, fcal)

    wtr_new = init.weather_parser(fwtr)
    x_new = init.feature_parser(wtr_new)

    reg = train.train()

    y_pew = reg.predict(x_new)
    pd.DataFrame(data=y_pew,
                 index=x_new.index,
                 columns=['prediction']).to_csv('output.csv')
    
    if fcal :
        y_new = init.y_parser(init.calls_parser(fcal))
        tester.printer(x_new, y_new, np.arange(len(x_new.index)), reg, binn='H')
    else :
        pd.DataFrame(data=y_new,
                     index=y_new.index,
                     columns=['prediction']).resample('').sum().plot()
        plt.title('Gradient Boosting Regressor')
        plt.show()

if __name__ == '__main__':
    main()
