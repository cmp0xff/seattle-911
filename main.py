
import sys

import init
import train

def main() :
    if len(sys.argv) == 1 or len(sys.argv) > 3 :
        print('usage: python main.py input.csv [calls.csv]')
    else :
        wtr_new = init.weather_parser(sys.argv[1])
        x_new = init.feature_parser(wtr_new)
        reg = train.train()
        y_pew = reg.predict(x_new)
        
        y_pew.to_csv('output.csv')
        
        if len(sys.argv) == 3 :
            calls_new = init.calls_parser()
            print('Score: ' + str(reg.score(x_new, y_new)))

if __name__ == '__main__':
    main()
