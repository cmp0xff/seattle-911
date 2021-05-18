
import pandas as pd

def init_calls() :
    # Convert to pandas DataFrame
    calls_df = pd.read_csv('./data/calls.csv')
    # print(calls_df.info())
    # print(calls_df.head(1))

    calls_df['Datetime'] = pd.to_datetime(calls_df['Datetime'],\
                                      format="%m/%d/%Y %I:%M:%S %p").dt.tz_localize(tz='US/Pacific', ambiguous='NaT')
    calls_df.dropna(inplace=True)
    calls_df.set_index('Datetime', inplace=True)
    calls_df.sort_index(inplace=True)

    # results_df['type'] = results_df['type'].astype('category')

    # print(calls_df.info())
    # print(calls_df.head(2))
    # print(calls_df.tail(2))
    
    return(calls_df)

def init_weather() :
    wtr_df = pd.read_csv('./data/Seattle Weatherdata 2002 to 2020.csv')
    wtr_df['datetime'] = pd.to_datetime(wtr_df['dt'], unit='s').\
    dt.tz_localize(tz='UTC').\
    dt.tz_convert('US/Pacific')
    wtr_df.set_index('datetime', inplace=True)

    # wtr_df.info()
    # wtr_df.head(2)
    
    return(wtr_df)

def init() :
    calls_df = init_calls()
    wtr_df = init_weather()
    
    x_raw = wtr_df[['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'weather_id']]
    # x_raw.drop_duplicates(inplace=True)

    y_raw = calls_df['Incident Number'].resample('H').count().to_frame('incident_count')
    xy_df = y_raw.join(x_raw.index.isocalendar())
    xy_df['hour'] = xy_df.index.hour
    xy_df = xy_df.join(x_raw).dropna()

    xy_df.drop_duplicates(inplace=True)

    # print(xy_df.info())

    return(xy_df)
