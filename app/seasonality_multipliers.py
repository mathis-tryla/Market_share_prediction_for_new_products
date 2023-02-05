import pandas as pd

# Get sales numbers per week
def get_sales_numbers_per_week_df(dataframe):
  sales_numbers_dict = {}
  for week in dataframe['week'].unique():
    sn = dataframe[dataframe['week'] == week].sales_number.values.sum()
    sales_numbers_dict[str(week)] = sn
  return sales_numbers_dict

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
    quarter_sales_numbers = dataframe.select((dataframe['week'] >= min_week_quarter) & (dataframe['week'] <= max_week_quarter))['sales_number'].sum()
    quarter_sales_numbers_total_dict = {"min": min_week_quarter,
                                        "max": max_week_quarter,
                                        "sales_number": quarter_sales_numbers}
    quarter_sales_numbers_total.append(quarter_sales_numbers_total_dict)
  return quarter_sales_numbers_total

# Convert sales numbers per quarter dict into dataframe
def get_sales_numbers_per_quarter_df(dataframe):
  sales_per_quarter = get_sales_numbers_per_quarter(dataframe)
  sales_per_quarter_dict = {}
  for i in range(len(sales_per_quarter)):
    quarter_sales = sales_per_quarter[i].get('sales_number')
    index = f"{sales_per_quarter[i].get('min')}-{sales_per_quarter[i].get('max')}"
    if quarter_sales == 0:
      sales_per_quarter_dict[index] = 0
    else:
      sales_per_quarter_dict[index] = quarter_sales
  return pd.DataFrame(sales_per_quarter_dict, index=[0])

"""
Get seasonality multipliers from 0 to 1
Note: is_per_week==False: get sales numbers per quarter
      is_per_week==True: get sales numbers per week
"""
def get_seasonality_multipliers(df, is_per_week=True):
  # Get the total number of sales stored in the dataset  
  total_sales_numbers = df['sales_number'].sum()
  print("-- Get the total number of sales stored in the dataframe DONE")

  seasonality_multipliers = {}
  if is_per_week:
    # Get the sales number per week
    df_sales_per_week = get_sales_numbers_per_week_df(df)
    print("-- Get the sales number per week DONE")
    # Calculate the part of sales numbers per week
    for (column_name, column_data) in df_sales_per_week.items():
      seasonality_multipliers[column_name] = column_data/total_sales_numbers
  else:
    # Get the sales number per quarter of 12 weeks or less
    df_sales_per_quarter = get_sales_numbers_per_quarter_df(df)
    print("-- Get the sales number per quarter of 12 weeks or less DONE")
    # Calculate the part of sales numbers per quarter
    for (column_name, column_data) in df_sales_per_quarter.items():
      seasonality_multipliers[column_name] = column_data.values/total_sales_numbers
  return seasonality_multipliers, total_sales_numbers