
import os
import pandas as pd
from time import process_time as timer

def calls_parser(fname) :
    print('Loading raw Seattle 911 calls database from ' + fname)
    tim = timer()
    calls_df = pd.read_csv(fname)
    print('Raw Seattle 911 calls database loaded in ' + str(timer() - tim))

    calls_df['Datetime'] = pd.to_datetime(calls_df['Datetime'],\
                                          format="%m/%d/%Y %I:%M:%S %p"\
                                         ).dt.tz_localize(tz='US/Pacific',\
                                                          ambiguous='NaT')

    calls_df.dropna(inplace=True)
    calls_df.set_index('Datetime', inplace=True)
    calls_df.index.set_names('datetime', inplace=True)
    calls_df.sort_index(inplace=True)

    return calls_df


def init_calls() :
    # Convert to pandas DataFrame
    
    calls_df_pt = './tmp/calls_df.parquet'
    
    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    if os.path.isfile(calls_df_pt) :
        print('Loading parsed Seattle 911 calls database from ' + str(calls_df_pt))
        tim = timer()
        calls_df = pd.read_parquet(calls_df_pt) # pd.read_hdf(calls_df_pt, key='c', mode='r')
        print('Parsed Seattle 911 calls database loaded in ' + str(timer() - tim) + ' s')
    else :
        calls_pt = './data/calls.csv'
    
        if not os.path.exists(calls_pt) :
            print('Downloading missing raw Seattle 911 calls database to ' + calls_pt)
            tim = timer()
            os.system('cat get_calls.sh | sh')
            print('Raw Seattle 911 calls database downloaded in ' + str(timer() - tim) + ' s')
    
        calls_df = calls_parser(calls_pt)

        print('Saving parsed Seattle 911 calls database to ' + calls_df_pt)
        tim = timer()
        calls_df.to_parquet(calls_df_pt) # calls_df.to_hdf(calls_df_pt, key='c', mode='w')
        print('Parsed Seattle 911 calls database saved in ' + str(timer() - tim) + ' s')

        # results_df['type'] = results_df['type'].astype('category')
    
    return calls_df

def weather_parser(fname) :
    print('Loading raw Seattle weather database from ' + fname)
    tim = timer()
    wtr_df = pd.read_csv(fname)
    print('Raw Seattle weather database loaded in ' + str(timer() - tim) + ' s')

    wtr_df['datetime'] = pd.to_datetime(wtr_df['dt'], unit='s').\
                            dt.tz_localize(tz='UTC').\
                            dt.tz_convert('US/Pacific')
    wtr_df.set_index('datetime', inplace=True)
    
    return(wtr_df)


def init_weather() :
    
    wtr_df_pt = './tmp/wtr_df.parquet'
    
    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    if os.path.isfile(wtr_df_pt) :
        print('Loading parsed Seattle weather database from ' + str(wtr_df_pt))
        tim = timer()
        wtr_df = pd.read_parquet(wtr_df_pt) # wtr_df = pd.read_hdf(wtr_df_pt, key='w', mode='r')
        # wtr_df.index.to_datetime().dt.tz_convert('US/Pacific')
        print('Parsed Seattle weather database loaded in ' + str(timer() - tim) + ' s')
    else :
        wtr_df = weather_parser('./data/Seattle Weatherdata 2002 to 2020.csv')

        wtr_df = wtr_df[~wtr_df.index.duplicated()]

        print('Saving parsed Seattle weather database to ' + wtr_df_pt)
        tim = timer()
        wtr_df.to_parquet(wtr_df_pt) # wtr_df.to_hdf(wtr_df_pt, key='w', mode='w')
        print('Parsed Seattle weather database saved in ' + str(timer() - tim) + ' s')

    return(wtr_df)

def feature_parser(wtr_df) :
    x = wtr_df[['temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'weather_id']]
    x.index.set_names('datetime', inplace=True)

    x_tim = x.index.isocalendar()
    x_tim['hour'] = x.index.hour
    x = x_tim.join(x)

    return x


def y_parser(calls_df) :
    y = calls_df['Incident Number'].resample('H').count().to_frame('incident_count')
    y.index.set_names('datetime', inplace=True)
    
    return y


def init() :
    
    xy_df_pt = './tmp/xy_df.parquet'
    
    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    if os.path.isfile(xy_df_pt) :
        print('Loading ML database from ' + str(xy_df_pt))
        tim = timer()
        xy_df = pd.read_parquet(xy_df_pt)
        print('ML database loaded in ' + str(timer() - tim) + ' s')
    else :
        
        tim = timer()
        calls_df = init_calls()
        wtr_df = init_weather()
        
        x_raw = feature_parser(wtr_df)

        # x_raw.drop_duplicates(inplace=True)

        y_raw = y_parser(calls_df)
        xy_df = y_raw.join(x_raw).dropna()
        # xy_df.index.set_names('datetime', inplace=True)
        # xy_df['hour'] = xy_df.index.hour

        xy_df.drop_duplicates(inplace=True)

        print('Saving ML database to ' + xy_df_pt)
        tim = timer()
        xy_df.to_parquet(xy_df_pt) #, key='x', mode='w'
        print('ML database saved in ' + str(timer() - tim) + ' s')

    return xy_df.iloc[:, 1:], xy_df.iloc[:, 0]
