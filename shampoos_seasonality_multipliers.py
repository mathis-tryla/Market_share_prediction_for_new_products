# How to run this script : 
# 1: sudo chmod <remove_bad_lines.sh> +x
# 2: python3 shampoos_seasonality_multipliers.py <remove_bad_lines.sh> <dataset.txt>

import pandas as pd
from io import StringIO, BytesIO
import math, time, os, sys 
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Get sales numbers per quarter
def get_sales_numbers_per_quarter(dataframe):
  min_week = dataframe['week'].min()
  max_week = dataframe['week'].max()
  print(min_week, max_week)
  quarters = []
  for i in range(min_week, max_week+1, 12):
    quarter = []
    for j in range(i, i+12):
      if j<=max_week:
        quarter.append(j)
    quarters.append(quarter)
  quarter_sales_numbers_total = []
  for quarter in quarters:
    min_week_quarter = min(quarter)
    max_week_quarter = max(quarter)
    quarter_sales_numbers = dataframe.loc[(dataframe['week'] >= min_week_quarter) & (dataframe['week'] <= max_week_quarter)]['sales_number'].sum()
    quarter_sales_numbers_total_dict = {"min": min_week_quarter,
                                        "max": max_week_quarter,
                                        "sales_numbers": quarter_sales_numbers}
    quarter_sales_numbers_total.append(quarter_sales_numbers_total_dict)
  return quarter_sales_numbers_total

# Convert sales numbers per quarter dict into dataframe
def get_sales_numbers_per_quarter_df(df):
  season_sales = 0
  sales_per_quarter = get_sales_numbers_per_quarter(df)
  sales_per_quarter_dict = {}
  for i in range(len(sales_per_quarter)):
    quarter_sales = sales_per_quarter[i].get('sales_numbers')
    index = f"{sales_per_quarter[i].get('min')}-{sales_per_quarter[i].get('max')}"
    if quarter_sales == 0:
      sales_per_quarter_dict[index] = 0
    else:
      sales_per_quarter_dict[index] = quarter_sales
  return pd.DataFrame(sales_per_quarter_dict, index=[0])

""" sys.argv[]
1 : Script which removes bad lines in the dataset
2 : Dataset file containing shampoos data
"""

def main():
  if len(sys.argv) > 1:
    remove_bad_lines_file = sys.argv[1] 
    dataset = sys.argv[2]

    # Remove lines which contains more fields than scheduled
    os.system(f"./{remove_bad_lines_file} {dataset}")

    # Create the dataframe
    headers = ['timestamp','week','barcode','type','segment','category','description','weight','sales_number','price','sales_value','discounts']
    df = pd.read_csv(dataset, sep=';', names=headers, index_col=False, encoding="utf-8", encoding_errors="ignore")

    # Get the sales number per quarter of 12 weeks or less
    df_sales_per_quarter = get_sales_numbers_per_quarter_df(df)

    # Get the total number of sales stored in the dataset  
    total_sales_numbers = df['sales_number'].sum()

    # Calculate the part of sales numbers per quarter
    seasonality_multipliers = {}
    for (column_name, column_data) in df_sales_per_quarter.items():
      seasonality_multipliers[column_name] = column_data.values/total_sales_numbers
  return seasonality_multipliers

start_time = time.time()
output = main()
end_time = time.time()
final_time = end_time - start_time
print(f"-- shampoos seasonality multipliers = {output} DONE")
print(f"-- {final_time} seconds--")
