# -*- coding: utf-8 -*-

import pandas as pd
from io import StringIO, BytesIO
import math, time, os, sys 
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm

def get_sales_per_timestamp(dataframe):
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
  winter_sales_numbers = dataframe.loc[(dataframe['week'] == 52) | ((dataframe['week'] >= 1) & (dataframe['week'] <= 11)), ['week', 'sales_numbers']].sales_numbers.values
  return {"spring": spring_sales_numbers, "summer": summer_sales_numbers, "fall": fall_sales_numbers, "winter": winter_sales_numbers}

# ventes moyennes de shampoings par semaine
def get_average_sales_per_week(df):
  if ('week' and 'sales_number') in df:
    return df.groupby('week')["sales_number"].mean()

# ventes moyenne de shampoings sur la période
def get_average_sales_per_season(df):
  # Get sales numbers per timestamp
  df_sales_numbers = get_sales_per_timestamp(df)
  df_sales_numbers = df_sales_numbers.sort_values('timestamp')

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
  #
  season_sales = 0
  sales_per_season = get_sales_per_season(df_sales_numbers)
  average_sales_per_season = {}
  for season in sales_per_season:
    for sale in sales_per_season[season]:
      season_sales += sale
    if len(sales_per_season[season]) == 0:
      average_sales_per_season[season] = 0
    else:
      average_sales_per_season[season] = (season_sales/len(sales_per_season[season]))
  return pd.DataFrame(average_sales_per_season, index=[0])

if __name__ == '__main__':
  # Start chrono
  start_time = time.time()

  """ sys.argv[]
  1 : script which removes bad line in the dataset
  2 : dataset file containing shampoos data
  3 : 
      0 is average shampoo sales per week
      1 is average shampoo sales over the seasons
  """
  #try:
  if len(sys.argv) > 1:
    remove_bad_lines_file = sys.argv[1] 
    dataset = sys.argv[2]
    #average_shampoo_sales_filter = sys.argv[3]

    os.system(f"./{remove_bad_lines_file} {dataset}")

    # Create the dataframe
    headers = ['timestamp','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discounts']
    df = pd.read_csv(dataset, sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore")

    df_average_sales_per_week = get_average_sales_per_week(df)
    #print(f"df_average_sales_per_week : {df_average_sales_per_week}")

    df_average_sales_per_season = get_average_sales_per_season(df)
    #print(f"df_average_sales_per_season : {df_average_sales_per_season}")

    total_sales_numbers = df['sales_number'].sum()

    # Calculate the part of sales numbers per season
    print("\nCOEFF per week:")
    for row in df_average_sales_per_week:
      print(row/total_sales_numbers)

    print("\nCOEFF per season:")
    print(f"spring={(df_average_sales_per_season['spring']/total_sales_numbers).values}")
    print(f"summer={(df_average_sales_per_season['summer']/total_sales_numbers).values}")
    print(f"fall={(df_average_sales_per_season['fall']/total_sales_numbers).values}")
    print(f"winter={(df_average_sales_per_season['winter']/total_sales_numbers).values}")
  
  #except:
  # End the chrono
  end_time = time.time()
  final_time = end_time - start_time
  print(f"-- {final_time} seconds--")

