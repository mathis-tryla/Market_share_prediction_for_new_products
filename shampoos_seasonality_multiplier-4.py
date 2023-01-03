# -*- coding: utf-8 -*-

import pandas as pd
from io import StringIO, BytesIO
import math, time, os, sys 
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm

def get_sales_per_timestamp(dataframe):
  curr_week = 0
  dict_sales_numbers = {}
  sales_numbers_list = []
  for timestamp in dataframe['timestamp'].unique():
    sales_numbers = dataframe.loc[dataframe['timestamp'] == timestamp, ['timestamp', 'barcode', 'sales_number']].sales_number.values.sum()
    sales_numbers_list.append(sales_numbers)
  dict_sales_numbers['timestamp'] = dataframe['timestamp'].unique()
  dict_sales_numbers['sales_numbers'] = sales_numbers_list
  return pd.DataFrame(dict_sales_numbers)

def get_date_from_timestamp(timestamp):
  date = datetime.fromtimestamp(timestamp)
  whole_date = {}
  whole_date["year"] = date.isocalendar()[0]
  whole_date["week"] = date.isocalendar()[1]
  whole_date["weekday"] = date.isocalendar()[2]
  return whole_date

def get_sales_per_season(dataframe):
  # Spring = 20/03-21/06 = 11-25
  spring = range(12, 26)
  # Summer = 21/06-23/09 = 26-38
  summer = range(26, 39)
  # Fall = 23/09-21/12 = 39-51
  fall = range(39, 52)
  # Winter = 21/12-20/03 = 52-11
  winter = range(0, 12) or range(52, 53)
  spring_sales_numbers = dataframe.loc[(dataframe['week'] >= 12) & (dataframe['week'] <= 25), ['week', 'sales_numbers']].sales_numbers.values
  summer_sales_numbers = dataframe.loc[(dataframe['week'] >= 26) & (dataframe['week'] <= 38), ['week', 'sales_numbers']].sales_numbers.values
  fall_sales_numbers = dataframe.loc[(dataframe['week'] >= 39) & (dataframe['week'] <= 51), ['week', 'sales_numbers']].sales_numbers.values
  winter_sales_numbers = dataframe.loc[(dataframe['week'] >= 52) & (dataframe['week'] < 53) & (dataframe['week'] >= 1) & (dataframe['week'] <= 11), ['week', 'sales_numbers']].sales_numbers.values
  return {"spring": spring_sales_numbers, "summer": summer_sales_numbers, "fall": fall_sales_numbers, "winter": winter_sales_numbers}


if __name__ == '__main__':
  # Start chrono
  start_time = time.time()

  if len(sys.argv) > 1:
    #remove_bad_lines_file = sys.argv[1] 
    dataset = sys.argv[1]

    #os.system(f"./{remove_bad_lines_file} {dataset}")

    # Create the dataframe
    headers = ['timestamp','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discounts']
    df = pd.read_csv(dataset, sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore")
    #df['timestamp'] = df['timestamp'].astype("str")
    #print(df.columns)
    #del df['Unnamed: 12']

    #df['timestamp'] = df['timestamp'].str[2:]
    df = df.sort_values('timestamp')
    print(df['timestamp'])
    print(df.head())
    print("dataframe created")

    # Get sales numbers per timestamp
    df_sales_numbers = get_sales_per_timestamp(df)
    df_sales_numbers = df_sales_numbers.sort_values('timestamp')
    print('df_sales_numbers')
    print(df_sales_numbers.head())

    # Create a dataframe for sales numbers with year, week and weekday per timestamp
    years = []
    weeks = []
    weekdays = []
    for ts in df_sales_numbers['timestamp']:
      timestamp = int(ts)
      years.append(get_date_from_timestamp(timestamp)['year'])
      weeks.append(get_date_from_timestamp(timestamp)['week'])
      weekdays.append(get_date_from_timestamp(timestamp)['weekday'])
    df_sales_numbers['year'] = years
    df_sales_numbers['week'] = weeks
    df_sales_numbers['weekday'] = weekdays
    print('df_sales_numbers')
    print(df_sales_numbers.head())

    # Get the season associated to each timestamp
    print("get sales per season")
    sales_per_season = get_sales_per_season(df_sales_numbers)

    # Test
    print(f"max week = {df_sales_numbers['week'].values.max()}")

    # Calculate the number of sales per season
    total_sales_numbers = 0
    total_sales_per_season = {}
    for season in sales_per_season:
      sn = 0
      for sales_numbers in sales_per_season[season]:
        sn += sales_numbers
      total_sales_per_season[season] = sn
      total_sales_numbers += sn
    print(total_sales_per_season['spring'])
    print(total_sales_per_season['summer'])
    print(total_sales_per_season['fall'])
    print(total_sales_per_season['winter'])

    # Calculate the part of sales numbers per season
    spring_multiplier = total_sales_per_season['spring']/total_sales_numbers
    summer_multiplier = total_sales_per_season['summer']/total_sales_numbers
    fall_multiplier = total_sales_per_season['fall']/total_sales_numbers
    winter_multiplier = total_sales_per_season['winter']/total_sales_numbers
    print(spring_multiplier, summer_multiplier, fall_multiplier, winter_multiplier)

    # End the chrono
    end_time = time.time()
    final_time = end_time - start_time
    print(f"-- {final_time} seconds--")

    # Plot the sales numbers per week
    df_sales_numbers.plot(x='week', y='sales_numbers', kind='line')
    plt.show()

  else:
    sys.exit(1)
