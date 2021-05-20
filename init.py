
import os
import pandas as pd

def init_calls() :
    # Convert to pandas DataFrame
    
    calls_df_pt = './tmp/calls_df.parquet'
    
    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    if os.path.isfile(calls_df_pt) :
        calls_df = pd.read_parquet(calls_df_pt) # pd.read_hdf(calls_df_pt, key='c', mode='r')
        # calls_df.index.to_datetime().dt.tz_convert('US/Pacific')
    else :
        calls_pt = './data/calls.csv'
    
        if not os.path.exists(calls_pt) :
            os.system('cat get_calls.sh | sh')
    
        calls_df = pd.read_csv(calls_pt)
        # print(calls_df.info())
        # print(calls_df.head(1))

        calls_df['Datetime'] = pd.to_datetime(calls_df['Datetime'],\
                                              format="%m/%d/%Y %I:%M:%S %p"\
                                             ).dt.tz_localize(tz='US/Pacific',\
                                                              ambiguous='NaT')
        calls_df.dropna(inplace=True)
        calls_df.set_index('Datetime', inplace=True)
        calls_df.index.set_names('datetime', inplace=True)
        calls_df.sort_index(inplace=True)

        calls_df.to_parquet(calls_df_pt) # calls_df.to_hdf(calls_df_pt, key='c', mode='w')

        # results_df['type'] = results_df['type'].astype('category')
    
    return(calls_df)

def weather_parser(fname) :
    wtr_df = pd.read_csv(fname)
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
        wtr_df = pd.read_parquet(wtr_df_pt) # wtr_df = pd.read_hdf(wtr_df_pt, key='w', mode='r')
        # wtr_df.index.to_datetime().dt.tz_convert('US/Pacific')
    else :
        wtr_df = weather_parser('./data/Seattle Weatherdata 2002 to 2020.csv')

        wtr_df = wtr_df[~wtr_df.index.duplicated()]

        wtr_df.to_parquet(wtr_df_pt) # wtr_df.to_hdf(wtr_df_pt, key='w', mode='w')

    return(wtr_df)

def ftr_parser(wtr_df) :
    x_raw = wtr_df[['temp', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'weather_id']]
    x_raw = x_raw.index.isocalendar().join(x_raw)
    x_raw.index.set_names('datetime', inplace=True)

    return x_raw

def init() :
    
    xy_df_pt = './tmp/xy_df.parquet'
    
    if not os.path.exists('./tmp') :
        os.system('mkdir tmp')
    if os.path.isfile(xy_df_pt) :
        xy_df = pd.read_parquet(xy_df_pt) # xy_df = pd.read_csv(xy_df_pt, index_col=0) #, key='x', mode='r'
        xy_df.index = pd.to_datetime(xy_df.index,
                                     format='%Y-%m-%d %H:%M:%S',
                                     utc=True).tz_convert('US/Pacific')
    else :
        calls_df = init_calls()
        wtr_df = init_weather()
        
        x_raw = ftr_parser(wtr_df)

        # x_raw.drop_duplicates(inplace=True)

        y_raw = calls_df['Incident Number'].resample('H').count().to_frame('incident_count')
        xy_df = y_raw.join(x_raw)
        xy_df.index.set_names('datetime', inplace=True)
        xy_df['hour'] = xy_df.index.hour
        xy_df = xy_df.join(x_raw).dropna()

        xy_df.drop_duplicates(inplace=True)

        xy_df.to_parquet(xy_df_pt) #, key='x', mode='w'

    return xy_df.iloc[:, 1:], xy_df.iloc[:, 0]
