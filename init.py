
import os
import pandas as pd

def init_calls() :
    # Convert to pandas DataFrame
    
    calls_df_pt = './tmp/calls_df.h5'
    
    if os.path.isfile(calls_df_pt) :
        return(pd.read_hdf(calls_df_pt, key='c', mode='r'))
    
    calls_pt = './data/calls.csv'
    
    if not os.path.exists(calls_pt) :
        os.system('cat get_calls.sh | sh')
    
    calls_df = pd.read_csv(calls_pt)
    # print(calls_df.info())
    # print(calls_df.head(1))

    calls_df['Datetime'] = pd.to_datetime(calls_df['Datetime'],\
                                      format="%m/%d/%Y %I:%M:%S %p").dt.tz_localize(tz='US/Pacific', ambiguous='NaT')
    calls_df.dropna(inplace=True)
    calls_df.set_index('Datetime', inplace=True)
    calls_df.sort_index(inplace=True)
    
    calls_df.to_hdf(calls_df_pt, key='c', mode='w')

    # results_df['type'] = results_df['type'].astype('category')
    
    return(calls_df)

def init_weather() :
    
    wtr_df_pt = './tmp/wtr_df.h5'
    
    if os.path.isfile(wtr_df_pt):
        return(pd.read_hdf(wtr_df_pt, key='w', mode='r'))
    
    wtr_df = pd.read_csv('./data/Seattle Weatherdata 2002 to 2020.csv')
    wtr_df['datetime'] = pd.to_datetime(wtr_df['dt'], unit='s').\
    dt.tz_localize(tz='UTC').\
    dt.tz_convert('US/Pacific')
    wtr_df.set_index('datetime', inplace=True)
    
    wtr_df.to_hdf(wtr_df_pt, key='w', mode='w')

    return(wtr_df)

def init() :
    
    xy_df_pt = './tmp/xy_df.csv'
    
    if os.path.isfile(xy_df_pt) :
        return(pd.read_csv(xy_df_pt)) #, key='x', mode='r'
    
    calls_df = init_calls()
    wtr_df = init_weather()
    
    x_raw = wtr_df[['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'weather_id']]
    # x_raw.drop_duplicates(inplace=True)

    y_raw = calls_df['Incident Number'].resample('H').count().to_frame('incident_count')
    xy_df = y_raw.join(x_raw.index.isocalendar())
    xy_df['hour'] = xy_df.index.hour
    xy_df = xy_df.join(x_raw).dropna()

    xy_df.drop_duplicates(inplace=True)
    
    print(xy_df.info())
    
    xy_df.to_csv(xy_df_pt) #, key='x', mode='w'

    return(xy_df)
